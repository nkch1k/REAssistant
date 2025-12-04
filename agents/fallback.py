"""Fallback agent for handling unclear or off-topic queries.

This module provides a fallback handler for queries that cannot be
processed by the main agents.
"""

import logging
from typing import Any

from state import AgentState

logger = logging.getLogger(__name__)


def fallback_node(state: AgentState) -> dict[str, Any]:
    """Fallback node for handling unclear or off-topic queries.

    This node provides helpful guidance when:
    - The query cannot be classified
    - The query is off-topic
    - There was an error in processing

    Args:
        state: Current agent state.

    Returns:
        State update dict with response field.

    Example:
        >>> state = {"user_query": "What's the weather?"}
        >>> result = fallback_node(state)
    """
    user_query = state.get("user_query", "")
    error = state.get("error")

    logger.info(f"Fallback handler triggered for query: {user_query}")

    # Build helpful response
    response = _build_fallback_response(user_query, error)

    return {
        "data": {},
        "response": response,
        "error": None,
    }


def _build_fallback_response(query: str, error: str | None) -> str:
    """Build a helpful fallback response.

    Args:
        query: The original user query.
        error: Optional error message.

    Returns:
        Formatted fallback response.
    """
    response = "I'm not sure how to help with that query.\n\n"

    response += "**I can help you with:**\n\n"
    response += "**Financial Analysis:**\n"
    response += '- "What\'s the total P&L for 2024?"\n'
    response += '- "Show me the expense breakdown"\n'
    response += '- "P&L for Q1 2024"\n\n'

    response += "**Property Queries:**\n"
    response += '- "How is Building 180 performing?"\n'
    response += '- "Compare Building 17 and Building 120"\n'
    response += '- "Show Building 140 revenue for 2024"\n\n'

    response += "**Tenant Analysis:**\n"
    response += '- "Revenue from Tenant 8"\n'
    response += '- "Top 5 tenants by revenue"\n'
    response += '- "How much does Tenant 12 pay?"\n\n'

    if error:
        response += f"_Technical note: {error}_\n"

    return response
