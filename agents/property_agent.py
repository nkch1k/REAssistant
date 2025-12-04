"""Property agent for property-related queries.

This module handles property details, comparisons, and performance queries.
"""

import logging
from typing import Any

from config import ERROR_PROPERTY_NOT_FOUND, Intent
from data.queries import (
    compare_properties,
    fuzzy_match_property,
    get_property_pnl,
    get_property_summary,
    get_worst_properties,
)
from state import AgentState

logger = logging.getLogger(__name__)


def property_agent_node(state: AgentState) -> dict[str, Any]:
    """Property agent node for handling property queries.

    This node processes property queries by:
    1. Extracting property names from entities
    2. Querying the data layer with fuzzy matching
    3. Formatting a user-friendly response

    Args:
        state: Current agent state with intent and entities.

    Returns:
        State update dict with data and response fields.

    Example:
        >>> state = {
        ...     "intent": "property_details",
        ...     "entities": {"property_name": "Building 180"}
        ... }
        >>> result = property_agent_node(state)
    """
    intent = state["intent"]
    entities = state["entities"]
    logger.info(f"Property agent processing intent: {intent}")

    try:
        if intent == Intent.PROPERTY_DETAILS.value:
            # Check if this is a ranking query (worst/best property)
            ranking_type = entities.get("ranking_type")
            entity_type = entities.get("entity_type")

            if ranking_type == "worst" and entity_type == "property":
                # Get worst performing property
                limit = entities.get("limit", 1)
                worst_props = get_worst_properties(n=limit)

                if not worst_props:
                    return {
                        "data": {},
                        "response": "No property data available.",
                        "error": None,
                    }

                data = {"worst_properties": worst_props}
                response = _format_worst_properties(worst_props)

            else:
                # Standard property details query
                property_name = entities.get("property_name")
                if not property_name:
                    return {
                        "data": {},
                        "response": "Please specify a property name.",
                        "error": "Missing property_name",
                    }

                # Check if property exists (with fuzzy matching)
                matched_name = fuzzy_match_property(property_name)
                if not matched_name:
                    return {
                        "data": {},
                        "response": f"{ERROR_PROPERTY_NOT_FOUND} Query: '{property_name}'",
                        "error": "Property not found",
                    }

                # Get property data
                year = entities.get("year")
                if year:
                    data = get_property_pnl(matched_name, year=year)
                else:
                    data = get_property_summary(matched_name)

                response = _format_property_details(data, matched_name, property_name)

        elif intent == Intent.PROPERTY_COMPARE.value:
            # Get comparison properties
            comparison_props = entities.get("comparison_properties", [])
            if len(comparison_props) < 2:
                # Try to extract from individual fields
                p1 = entities.get("property_1") or entities.get("property_name")
                p2 = entities.get("property_2")
                if p1 and p2:
                    comparison_props = [p1, p2]
                else:
                    return {
                        "data": {},
                        "response": "Please specify two properties to compare.",
                        "error": "Missing properties for comparison",
                    }

            p1, p2 = comparison_props[0], comparison_props[1]

            # Fuzzy match both properties
            matched_p1 = fuzzy_match_property(p1)
            matched_p2 = fuzzy_match_property(p2)

            if not matched_p1:
                return {
                    "data": {},
                    "response": f"{ERROR_PROPERTY_NOT_FOUND} Query: '{p1}'",
                    "error": "Property 1 not found",
                }

            if not matched_p2:
                return {
                    "data": {},
                    "response": f"{ERROR_PROPERTY_NOT_FOUND} Query: '{p2}'",
                    "error": "Property 2 not found",
                }

            # Compare properties
            data = compare_properties(matched_p1, matched_p2)
            response = _format_property_comparison(data)

        else:
            logger.warning(f"Unexpected intent in property agent: {intent}")
            return {
                "data": {},
                "response": "Unable to process property query.",
                "error": f"Unexpected intent: {intent}",
            }

        logger.info("Property agent completed successfully")
        return {
            "data": data,
            "response": response,
            "error": None,
        }

    except ValueError as e:
        logger.error(f"Value error in property agent: {e}")
        return {
            "data": {},
            "response": str(e),
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error in property agent: {e}")
        return {
            "data": {},
            "response": f"Error processing property query: {str(e)}",
            "error": str(e),
        }


def _format_property_details(
    data: dict, matched_name: str, original_query: str
) -> str:
    """Format property details response.

    Args:
        data: Property data dictionary.
        matched_name: The matched property name.
        original_query: Original query from user.

    Returns:
        Formatted response string.
    """
    response = ""

    # Add fuzzy match notice if needed
    if matched_name.lower() != original_query.lower():
        response += f"_Showing results for: {matched_name}_\n\n"

    response += f"**{matched_name}**\n\n"

    # Check if we have P&L data or summary
    if "revenue" in data and "expenses" in data:
        # P&L breakdown
        revenue = data["revenue"]
        expenses = abs(data["expenses"])
        net_profit = data["net_profit"]
        year = data.get("year")

        period = f" ({year})" if year else ""
        response += f"**Financial Performance{period}:**\n"
        response += f"- Revenue: ${revenue:,.2f}\n"
        response += f"- Expenses: ${expenses:,.2f}\n"
        response += f"- Net {'Profit' if net_profit > 0 else 'Loss'}: ${abs(net_profit):,.2f}\n"

    elif "total_revenue" in data:
        # Summary data
        total_pnl = data["total_pnl"]
        total_revenue = data["total_revenue"]
        total_expenses = abs(data["total_expenses"])
        tenant_count = data["tenant_count"]

        response += "**Overall Performance:**\n"
        response += f"- Total Revenue: ${total_revenue:,.2f}\n"
        response += f"- Total Expenses: ${total_expenses:,.2f}\n"
        response += f"- Net {'Profit' if total_pnl > 0 else 'Loss'}: ${abs(total_pnl):,.2f}\n"
        response += f"- Tenants: {tenant_count}\n"

        if tenant_count > 0 and "tenants" in data:
            tenants = data["tenants"]
            if len(tenants) <= 5:
                response += f"  - {', '.join(tenants)}\n"
            else:
                response += f"  - {', '.join(tenants[:5])}, and {len(tenants) - 5} more\n"

    return response


def _format_property_comparison(data: dict) -> str:
    """Format property comparison response.

    Args:
        data: Comparison data dictionary.

    Returns:
        Formatted response string.
    """
    prop1 = data["property_1"]
    prop2 = data["property_2"]
    diff = data["difference"]

    response = "**Property Comparison**\n\n"

    # Property 1
    response += f"**{prop1['property_name']}:**\n"
    response += f"- Total P&L: ${prop1['total_pnl']:,.2f}\n"
    response += f"- Revenue: ${prop1['total_revenue']:,.2f}\n"
    response += f"- Expenses: ${abs(prop1['total_expenses']):,.2f}\n"
    response += f"- Tenants: {prop1['tenant_count']}\n\n"

    # Property 2
    response += f"**{prop2['property_name']}:**\n"
    response += f"- Total P&L: ${prop2['total_pnl']:,.2f}\n"
    response += f"- Revenue: ${prop2['total_revenue']:,.2f}\n"
    response += f"- Expenses: ${abs(prop2['total_expenses']):,.2f}\n"
    response += f"- Tenants: {prop2['tenant_count']}\n\n"

    # Difference
    response += "**Difference (Property 1 - Property 2):**\n"
    response += f"- P&L: ${diff['total_pnl']:+,.2f}\n"
    response += f"- Revenue: ${diff['total_revenue']:+,.2f}\n"
    response += f"- Expenses: ${diff['total_expenses']:+,.2f}\n"

    return response


def _format_worst_properties(worst_props: list[dict]) -> str:
    """Format worst properties response.

    Args:
        worst_props: List of worst property dictionaries.

    Returns:
        Formatted response string.
    """
    if len(worst_props) == 1:
        prop = worst_props[0]
        response = f"**Worst Performing Property**\n\n"
        response += f"**{prop['property_name']}**\n"
        response += f"- Total P&L: ${prop['total_pnl']:,.2f}\n"
        if prop['total_pnl'] < 0:
            response += f"\nThis property is operating at a loss."
        else:
            response += f"\nThis property has the lowest profit among all properties."
    else:
        response = f"**Bottom {len(worst_props)} Properties by P&L**\n\n"
        for prop in worst_props:
            response += f"{prop['rank']}. **{prop['property_name']}**: ${prop['total_pnl']:,.2f}\n"

    return response
