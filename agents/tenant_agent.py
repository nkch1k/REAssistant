"""Tenant agent for tenant-related queries.

This module handles tenant revenue queries and rankings.
"""

import logging
from typing import Any

from config import ERROR_TENANT_NOT_FOUND, Intent
from data.queries import (
    fuzzy_match_tenant,
    get_tenant_revenue,
    get_top_tenants,
    get_worst_tenants,
)
from state import AgentState

logger = logging.getLogger(__name__)


def tenant_agent_node(state: AgentState) -> dict[str, Any]:
    """Tenant agent node for handling tenant queries.

    This node processes tenant queries by:
    1. Extracting tenant names from entities
    2. Querying the data layer with fuzzy matching
    3. Formatting a user-friendly response

    Args:
        state: Current agent state with intent and entities.

    Returns:
        State update dict with data and response fields.

    Example:
        >>> state = {
        ...     "intent": "tenant_details",
        ...     "entities": {"tenant_name": "Tenant 8"}
        ... }
        >>> result = tenant_agent_node(state)
    """
    intent = state["intent"]
    entities = state["entities"]
    logger.info(f"Tenant agent processing intent: {intent}")

    try:
        if intent == Intent.TENANT_DETAILS.value:
            # Get tenant name
            tenant_name = entities.get("tenant_name")
            if not tenant_name:
                return {
                    "data": {},
                    "response": "Please specify a tenant name.",
                    "error": "Missing tenant_name",
                }

            # Check if tenant exists (with fuzzy matching)
            matched_name = fuzzy_match_tenant(tenant_name)
            if not matched_name:
                return {
                    "data": {},
                    "response": f"{ERROR_TENANT_NOT_FOUND} Query: '{tenant_name}'",
                    "error": "Tenant not found",
                }

            # Get tenant revenue
            year = entities.get("year")
            revenue = get_tenant_revenue(matched_name, year=year)

            data = {
                "tenant_name": matched_name,
                "revenue": revenue,
                "year": year,
            }

            response = _format_tenant_details(
                matched_name, revenue, year, tenant_name
            )

        elif intent == Intent.TENANT_RANKING.value:
            # Get limit
            limit = entities.get("limit", 5)
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                limit = 5

            # Check if worst ranking
            ranking_type = entities.get("ranking_type", "best")

            if ranking_type == "worst":
                # Get worst tenants
                tenants = get_worst_tenants(n=limit)
                if not tenants:
                    return {
                        "data": {},
                        "response": "No tenant data available.",
                        "error": None,
                    }
                data = {"worst_tenants": tenants, "limit": limit}
                response = _format_worst_tenants(tenants)
            else:
                # Get top tenants
                tenants = get_top_tenants(n=limit)
                if not tenants:
                    return {
                        "data": {},
                        "response": "No tenant data available.",
                        "error": None,
                    }
                data = {"top_tenants": tenants, "limit": limit}
                response = _format_tenant_ranking(tenants)

        else:
            logger.warning(f"Unexpected intent in tenant agent: {intent}")
            return {
                "data": {},
                "response": "Unable to process tenant query.",
                "error": f"Unexpected intent: {intent}",
            }

        logger.info("Tenant agent completed successfully")
        return {
            "data": data,
            "response": response,
            "error": None,
        }

    except ValueError as e:
        logger.error(f"Value error in tenant agent: {e}")
        return {
            "data": {},
            "response": str(e),
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error in tenant agent: {e}")
        return {
            "data": {},
            "response": f"Error processing tenant query: {str(e)}",
            "error": str(e),
        }


def _format_tenant_details(
    matched_name: str, revenue: float, year: str | None, original_query: str
) -> str:
    """Format tenant details response.

    Args:
        matched_name: The matched tenant name.
        revenue: Total revenue from tenant.
        year: Year filter if applicable.
        original_query: Original query from user.

    Returns:
        Formatted response string.
    """
    response = ""

    # Add fuzzy match notice if needed
    if matched_name.lower() != original_query.lower():
        response += f"_Showing results for: {matched_name}_\n\n"

    response += f"**{matched_name}**\n\n"

    period = f" in {year}" if year else " (all time)"
    response += f"Total Revenue{period}: ${revenue:,.2f}\n"

    return response


def _format_tenant_ranking(top_tenants: list[dict]) -> str:
    """Format tenant ranking response.

    Args:
        top_tenants: List of top tenant dictionaries.

    Returns:
        Formatted response string.
    """
    response = f"**Top {len(top_tenants)} Tenants by Revenue**\n\n"

    for tenant in top_tenants:
        rank = tenant["rank"]
        name = tenant["tenant_name"]
        revenue = tenant["total_revenue"]

        response += f"{rank}. **{name}**: ${revenue:,.2f}\n"

    return response


def _format_worst_tenants(worst_tenants: list[dict]) -> str:
    """Format worst tenants response.

    Args:
        worst_tenants: List of worst tenant dictionaries.

    Returns:
        Formatted response string.
    """
    if len(worst_tenants) == 1:
        tenant = worst_tenants[0]
        response = f"**Worst Performing Tenant**\n\n"
        response += f"**{tenant['tenant_name']}**\n"
        response += f"- Total Revenue: ${tenant['total_revenue']:,.2f}\n"
        response += f"\nThis tenant has the lowest revenue among all tenants."
    else:
        response = f"**Bottom {len(worst_tenants)} Tenants by Revenue**\n\n"
        for tenant in worst_tenants:
            response += f"{tenant['rank']}. **{tenant['tenant_name']}**: ${tenant['total_revenue']:,.2f}\n"

    return response
