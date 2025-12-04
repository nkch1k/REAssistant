"""State definition for LangGraph workflow.

This module defines the TypedDict state used throughout the LangGraph
workflow for tracking user queries, intents, entities, and responses.
"""

from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """State schema for the multi-agent workflow.

    Attributes:
        user_query: The original user question or request.
        intent: Classified intent type (pnl | property | tenant | fallback).
        entities: Dictionary of extracted entities (properties, tenants, timeframes).
        data: Dictionary containing query results from data layer.
        response: Final formatted response to return to user.
        error: Optional error message if something went wrong.
    """

    user_query: str
    intent: str
    entities: dict[str, Any]
    data: dict[str, Any]
    response: str
    error: Optional[str]
