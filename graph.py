"""LangGraph workflow definition.

This module defines the multi-agent workflow using LangGraph's StateGraph.
The workflow routes queries through appropriate agents based on intent.
"""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agents.fallback import fallback_node
from agents.pnl_agent import pnl_agent_node
from agents.property_agent import property_agent_node
from agents.router import route_intent, router_node
from agents.tenant_agent import tenant_agent_node
from state import AgentState

logger = logging.getLogger(__name__)


def create_graph() -> StateGraph:
    """Create and configure the LangGraph workflow.

    This function builds the multi-agent workflow with the following structure:
    1. Router node classifies intent and extracts entities
    2. Conditional routing to appropriate agent (P&L, property, tenant, or fallback)
    3. Agent processes query and returns response

    Returns:
        Configured StateGraph instance ready for compilation.

    Example:
        >>> graph = create_graph()
        >>> app = graph.compile()
        >>> result = app.invoke({"user_query": "What's the P&L for 2024?"})
    """
    # Initialize state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("pnl_agent", pnl_agent_node)
    workflow.add_node("property_agent", property_agent_node)
    workflow.add_node("tenant_agent", tenant_agent_node)
    workflow.add_node("fallback", fallback_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_intent,
        {
            "pnl_agent": "pnl_agent",
            "property_agent": "property_agent",
            "tenant_agent": "tenant_agent",
            "fallback": "fallback",
        },
    )

    # Add edges from agents to END
    workflow.add_edge("pnl_agent", END)
    workflow.add_edge("property_agent", END)
    workflow.add_edge("tenant_agent", END)
    workflow.add_edge("fallback", END)

    logger.info("Graph created successfully")
    return workflow


def run_query(query: str) -> dict[str, Any]:
    """Run a query through the multi-agent workflow.

    This is a convenience function that:
    1. Creates the graph
    2. Compiles it
    3. Runs the query
    4. Returns the final state

    Args:
        query: User query string.

    Returns:
        Final state dictionary with response.

    Example:
        >>> result = run_query("What's the P&L for 2024?")
        >>> print(result["response"])
    """
    logger.info(f"Running query: {query}")

    # Create and compile graph
    workflow = create_graph()
    app = workflow.compile()

    # Initialize state
    initial_state: AgentState = {
        "user_query": query,
        "intent": "",
        "entities": {},
        "data": {},
        "response": "",
        "error": None,
    }

    # Run workflow
    try:
        final_state = app.invoke(initial_state)
        logger.info("Query completed successfully")
        return final_state
    except Exception as e:
        logger.error(f"Error running query: {e}")
        return {
            "user_query": query,
            "intent": "fallback",
            "entities": {},
            "data": {},
            "response": f"An error occurred: {str(e)}",
            "error": str(e),
        }
