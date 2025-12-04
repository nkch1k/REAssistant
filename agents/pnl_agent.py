"""P&L agent for profit and loss queries.

This module handles all P&L related queries including summaries and
breakdowns by period.
"""

import logging
from typing import Any

from config import ERROR_NO_DATA, Intent
from data.queries import get_pnl_breakdown, get_total_pnl
from state import AgentState

logger = logging.getLogger(__name__)


def pnl_agent_node(state: AgentState) -> dict[str, Any]:
    """P&L agent node for handling profit and loss queries.

    This node processes P&L queries by:
    1. Extracting relevant entities (year, quarter)
    2. Querying the data layer
    3. Formatting a user-friendly response

    Args:
        state: Current agent state with intent and entities.

    Returns:
        State update dict with data and response fields.

    Example:
        >>> state = {
        ...     "intent": "pnl_summary",
        ...     "entities": {"year": "2024"}
        ... }
        >>> result = pnl_agent_node(state)
    """
    intent = state["intent"]
    entities = state["entities"]
    logger.info(f"P&L agent processing intent: {intent}")

    try:
        # Extract entities
        year = entities.get("year")
        quarter = entities.get("quarter")

        # Query data based on intent
        if intent == Intent.PNL_SUMMARY.value:
            total_pnl = get_total_pnl(year=year, quarter=quarter)

            if total_pnl == 0:
                return {
                    "data": {},
                    "response": ERROR_NO_DATA,
                    "error": None,
                }

            data = {
                "total_pnl": total_pnl,
                "year": year,
                "quarter": quarter,
            }

            # Format response
            period = quarter if quarter else (f"year {year}" if year else "all periods")
            response = _format_pnl_summary(total_pnl, period)

        elif intent == Intent.PNL_BREAKDOWN.value:
            breakdown = get_pnl_breakdown(year=year)

            if not breakdown:
                return {
                    "data": {},
                    "response": ERROR_NO_DATA,
                    "error": None,
                }

            data = {
                "breakdown": breakdown,
                "year": year,
            }

            # Format response
            response = _format_pnl_breakdown(breakdown, year)

        else:
            logger.warning(f"Unexpected intent in P&L agent: {intent}")
            return {
                "data": {},
                "response": "Unable to process P&L query.",
                "error": f"Unexpected intent: {intent}",
            }

        logger.info(f"P&L agent completed successfully")
        return {
            "data": data,
            "response": response,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error in P&L agent: {e}")
        return {
            "data": {},
            "response": f"Error processing P&L query: {str(e)}",
            "error": str(e),
        }


def _format_pnl_summary(total_pnl: float, period: str) -> str:
    """Format P&L summary response.

    Args:
        total_pnl: Total profit/loss value.
        period: Time period description.

    Returns:
        Formatted response string.
    """
    # Determine if profit or loss
    status = "profit" if total_pnl > 0 else "loss"

    response = f"**P&L Summary for {period}**\n\n"
    response += f"Total: ${abs(total_pnl):,.2f} ({status})\n"

    return response


def _format_pnl_breakdown(breakdown: dict[str, float], year: str | None) -> str:
    """Format P&L breakdown response.

    Args:
        breakdown: Dictionary of ledger groups and amounts.
        year: Year filter if applicable.

    Returns:
        Formatted response string.
    """
    period = f"year {year}" if year else "all periods"

    response = f"**P&L Breakdown for {period}**\n\n"

    # Separate revenue and expenses
    revenue_items = []
    expense_items = []

    for category, amount in breakdown.items():
        if amount > 0:
            revenue_items.append((category, amount))
        else:
            expense_items.append((category, abs(amount)))

    # Sort by amount
    revenue_items.sort(key=lambda x: x[1], reverse=True)
    expense_items.sort(key=lambda x: x[1], reverse=True)

    # Format revenue
    if revenue_items:
        response += "**Revenue:**\n"
        total_revenue = sum(amt for _, amt in revenue_items)
        for category, amount in revenue_items:
            response += f"- {category.replace('_', ' ').title()}: ${amount:,.2f}\n"
        response += f"- **Total Revenue: ${total_revenue:,.2f}**\n\n"

    # Format expenses
    if expense_items:
        response += "**Expenses:**\n"
        total_expenses = sum(amt for _, amt in expense_items)
        for category, amount in expense_items:
            response += f"- {category.replace('_', ' ').title()}: ${amount:,.2f}\n"
        response += f"- **Total Expenses: ${total_expenses:,.2f}**\n\n"

    # Net profit/loss
    total_pnl = sum(breakdown.values())
    status = "Profit" if total_pnl > 0 else "Loss"
    response += f"**Net {status}: ${abs(total_pnl):,.2f}**"

    return response
