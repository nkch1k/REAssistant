"""Query functions for real estate data.

This module provides all query functions for retrieving and analyzing
real estate data from the parquet dataset.
"""

import logging
from typing import Any, Optional

import pandas as pd
from rapidfuzz import fuzz, process

from config import FUZZY_MATCH_THRESHOLD, LedgerType
from data.loader import get_dataframe

logger = logging.getLogger(__name__)


def get_total_pnl(year: Optional[str] = None, quarter: Optional[str] = None) -> float:
    """Get total profit and loss for a given period.

    Args:
        year: Year filter (e.g., "2024"). If None, all years included.
        quarter: Quarter filter (e.g., "2024-Q1"). If None, all quarters included.

    Returns:
        Total P&L as float. Positive means profit, negative means loss.

    Example:
        >>> total = get_total_pnl(year="2024")
        >>> quarterly = get_total_pnl(quarter="2024-Q1")
    """
    df = get_dataframe()

    if quarter:
        df = df[df["quarter"] == quarter]
    elif year:
        df = df[df["year"] == year]

    total = df["profit"].sum()
    logger.info(f"Total P&L (year={year}, quarter={quarter}): {total}")
    return float(total)


def get_pnl_breakdown(year: Optional[str] = None) -> dict[str, float]:
    """Get P&L breakdown by ledger group.

    Args:
        year: Year filter (e.g., "2024"). If None, all years included.

    Returns:
        Dictionary mapping ledger group names to their totals.

    Example:
        >>> breakdown = get_pnl_breakdown(year="2024")
        >>> print(breakdown["rental_income"])
    """
    df = get_dataframe()

    if year:
        df = df[df["year"] == year]

    breakdown = df.groupby("ledger_group")["profit"].sum().to_dict()
    logger.info(f"P&L breakdown for year={year}: {len(breakdown)} groups")
    return {str(k): float(v) for k, v in breakdown.items()}


def get_property_summary(name: str) -> dict:
    """Get comprehensive summary for a property.

    Args:
        name: Property name (e.g., "Building 180").

    Returns:
        Dictionary containing property statistics.

    Raises:
        ValueError: If property not found.

    Example:
        >>> summary = get_property_summary("Building 180")
        >>> print(summary["total_pnl"])
    """
    # Try fuzzy match first
    matched_name = fuzzy_match_property(name)
    if not matched_name:
        raise ValueError(f"Property '{name}' not found")

    df = get_dataframe()
    prop_df = df[df["property_name"] == matched_name]

    if prop_df.empty:
        raise ValueError(f"No data for property '{matched_name}'")

    # Get unique tenants, filtering out None values
    tenants_list = [t for t in prop_df["tenant_name"].unique().tolist() if t is not None]

    summary = {
        "property_name": matched_name,
        "total_pnl": float(prop_df["profit"].sum()),
        "total_revenue": float(prop_df[prop_df["ledger_type"] == LedgerType.REVENUE.value]["profit"].sum()),
        "total_expenses": float(prop_df[prop_df["ledger_type"] == LedgerType.EXPENSES.value]["profit"].sum()),
        "tenant_count": int(prop_df["tenant_name"].nunique()),
        "tenants": sorted(tenants_list),
    }

    logger.info(f"Property summary for {matched_name}: P&L={summary['total_pnl']}")
    return summary


def get_property_pnl(name: str, year: Optional[str] = None) -> dict:
    """Get P&L for a specific property with optional year filter.

    Args:
        name: Property name (e.g., "Building 180").
        year: Year filter (e.g., "2024"). If None, all years included.

    Returns:
        Dictionary containing P&L breakdown.

    Raises:
        ValueError: If property not found.

    Example:
        >>> pnl = get_property_pnl("Building 180", year="2024")
    """
    matched_name = fuzzy_match_property(name)
    if not matched_name:
        raise ValueError(f"Property '{name}' not found")

    df = get_dataframe()
    prop_df = df[df["property_name"] == matched_name]

    if year:
        prop_df = prop_df[prop_df["year"] == year]

    if prop_df.empty:
        raise ValueError(f"No data for property '{matched_name}' in year {year}")

    revenue_df = prop_df[prop_df["ledger_type"] == LedgerType.REVENUE.value]
    expense_df = prop_df[prop_df["ledger_type"] == LedgerType.EXPENSES.value]

    result = {
        "property_name": matched_name,
        "year": year,
        "revenue": float(revenue_df["profit"].sum()),
        "expenses": float(expense_df["profit"].sum()),
        "net_profit": float(prop_df["profit"].sum()),
        "revenue_breakdown": revenue_df.groupby("ledger_group")["profit"].sum().to_dict(),
        "expense_breakdown": expense_df.groupby("ledger_group")["profit"].sum().to_dict(),
    }

    # Convert breakdown dicts
    result["revenue_breakdown"] = {str(k): float(v) for k, v in result["revenue_breakdown"].items()}
    result["expense_breakdown"] = {str(k): float(v) for k, v in result["expense_breakdown"].items()}

    logger.info(f"Property P&L for {matched_name} (year={year}): {result['net_profit']}")
    return result


def get_tenant_revenue(name: str, year: Optional[str] = None) -> float:
    """Get total revenue from a specific tenant.

    Args:
        name: Tenant name (e.g., "Tenant 8").
        year: Year filter (e.g., "2024"). If None, all years included.

    Returns:
        Total revenue as float.

    Raises:
        ValueError: If tenant not found.

    Example:
        >>> revenue = get_tenant_revenue("Tenant 8", year="2024")
    """
    matched_name = fuzzy_match_tenant(name)
    if not matched_name:
        raise ValueError(f"Tenant '{name}' not found")

    df = get_dataframe()
    tenant_df = df[df["tenant_name"] == matched_name]
    tenant_df = tenant_df[tenant_df["ledger_type"] == LedgerType.REVENUE.value]

    if year:
        tenant_df = tenant_df[tenant_df["year"] == year]

    if tenant_df.empty:
        return 0.0

    revenue = tenant_df["profit"].sum()
    logger.info(f"Tenant {matched_name} revenue (year={year}): {revenue}")
    return float(revenue)


def fuzzy_match_property(query: str) -> Optional[str]:
    """Find the best matching property name using fuzzy matching.

    Args:
        query: Property name query (can be partial or misspelled).

    Returns:
        Best matching property name, or None if no good match found.

    Example:
        >>> matched = fuzzy_match_property("bldg 180")
        >>> print(matched)  # "Building 180"
    """
    df = get_dataframe()
    property_names = df["property_name"].unique().tolist()

    if query in property_names:
        return query

    result = process.extractOne(
        query,
        property_names,
        scorer=fuzz.ratio,
        score_cutoff=FUZZY_MATCH_THRESHOLD,
    )

    if result:
        matched_name, score, _ = result
        logger.info(f"Fuzzy matched '{query}' to '{matched_name}' (score: {score})")
        return str(matched_name)

    logger.warning(f"No fuzzy match found for property '{query}'")
    return None


def fuzzy_match_tenant(query: str) -> Optional[str]:
    """Find the best matching tenant name using fuzzy matching.

    Args:
        query: Tenant name query (can be partial or misspelled).

    Returns:
        Best matching tenant name, or None if no good match found.

    Example:
        >>> matched = fuzzy_match_tenant("tenant8")
        >>> print(matched)  # "Tenant 8"
    """
    df = get_dataframe()
    tenant_names = df["tenant_name"].unique().tolist()

    if query in tenant_names:
        return query

    result = process.extractOne(
        query,
        tenant_names,
        scorer=fuzz.ratio,
        score_cutoff=FUZZY_MATCH_THRESHOLD,
    )

    if result:
        matched_name, score, _ = result
        logger.info(f"Fuzzy matched '{query}' to '{matched_name}' (score: {score})")
        return str(matched_name)

    logger.warning(f"No fuzzy match found for tenant '{query}'")
    return None


def get_all_properties_with_pnl() -> list[dict]:
    """Get all properties with their P&L data sorted by P&L descending.

    Returns:
        List of dictionaries with property info and P&L, sorted best to worst.

    Example:
        >>> all_props = get_all_properties_with_pnl()
        >>> print(f"Best: {all_props[0]['property_name']}")
    """
    df = get_dataframe()

    # Calculate P&L for each property
    property_pnl = df.groupby("property_name")["profit"].sum().sort_values(ascending=False)

    # Get additional details for each property
    result = []
    for prop_name, total_pnl in property_pnl.items():
        prop_df = df[df["property_name"] == prop_name]
        revenue_df = prop_df[prop_df["ledger_type"] == LedgerType.REVENUE.value]
        expense_df = prop_df[prop_df["ledger_type"] == LedgerType.EXPENSES.value]

        result.append({
            "property_name": prop_name,
            "total_pnl": float(total_pnl),
            "total_revenue": float(revenue_df["profit"].sum()),
            "total_expenses": float(expense_df["profit"].sum()),
            "tenant_count": int(prop_df["tenant_name"].nunique()),
        })

    logger.info(f"Retrieved all {len(result)} properties with P&L")
    return result


def get_all_tenants_with_revenue() -> list[dict]:
    """Get all tenants with their revenue data sorted by revenue descending.

    Returns:
        List of dictionaries with tenant info and revenue, sorted best to worst.

    Example:
        >>> all_tenants = get_all_tenants_with_revenue()
        >>> print(f"Best: {all_tenants[0]['tenant_name']}")
    """
    df = get_dataframe()
    revenue_df = df[df["ledger_type"] == LedgerType.REVENUE.value]

    # Calculate revenue for each tenant
    tenant_revenue = revenue_df.groupby("tenant_name")["profit"].sum().sort_values(ascending=False)

    result = [
        {
            "tenant_name": tenant,
            "total_revenue": float(revenue),
            "rank": idx + 1,
        }
        for idx, (tenant, revenue) in enumerate(tenant_revenue.items())
    ]

    logger.info(f"Retrieved all {len(result)} tenants with revenue")
    return result


def get_portfolio_stats() -> dict[str, Any]:
    """Get comprehensive portfolio statistics.

    Returns:
        Dictionary containing overall portfolio metrics.

    Example:
        >>> stats = get_portfolio_stats()
        >>> print(f"Properties: {stats['property_count']}")
    """
    df = get_dataframe()

    # Get revenue and expense data
    revenue_df = df[df["ledger_type"] == LedgerType.REVENUE.value]
    expense_df = df[df["ledger_type"] == LedgerType.EXPENSES.value]

    # Filter out None values and convert to strings for years
    years = [str(int(y)) for y in df["year"].unique() if pd.notna(y)]
    properties = [p for p in df["property_name"].unique() if pd.notna(p)]
    tenants = [t for t in df["tenant_name"].unique() if pd.notna(t)]

    stats = {
        "property_count": int(df["property_name"].nunique()),
        "tenant_count": int(df["tenant_name"].nunique()),
        "properties": sorted(properties),
        "tenants": sorted(tenants),
        "total_revenue": float(revenue_df["profit"].sum()),
        "total_expenses": float(expense_df["profit"].sum()),
        "net_pnl": float(df["profit"].sum()),
        "years_covered": sorted(years),
    }

    logger.info(f"Portfolio stats: {stats['property_count']} properties, {stats['tenant_count']} tenants")
    return stats
