"""Router agent for intent classification and entity extraction.

This module contains the router node that classifies user intent and
extracts relevant entities from the query.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, Intent
from prompts.templates import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from state import AgentState

logger = logging.getLogger(__name__)


def router_node(state: AgentState) -> dict[str, Any]:
    """Router node that classifies intent and extracts entities.

    This node uses an LLM to analyze the user query and determine:
    1. The intent type (P&L, property, tenant, or fallback)
    2. Relevant entities (property names, tenant names, dates)

    Args:
        state: Current agent state containing user_query.

    Returns:
        State update dict with intent and entities fields.

    Example:
        >>> state = {"user_query": "What's the P&L for 2024?"}
        >>> result = router_node(state)
        >>> print(result["intent"])  # "pnl_summary"
    """
    user_query = state["user_query"]
    logger.info(f"Router processing query: {user_query}")

    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
        )

        # Create messages
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=ROUTER_USER_TEMPLATE.format(query=user_query)),
        ]

        # Get LLM response
        response = llm.invoke(messages)
        response_text = response.content

        logger.info(f"Router LLM response: {response_text}")

        # Parse JSON response
        try:
            parsed = json.loads(response_text)
            intent = parsed.get("intent", Intent.FALLBACK.value)
            entities = parsed.get("entities", {})
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse router response as JSON: {e}")
            intent = Intent.FALLBACK.value
            entities = {}

        # Validate intent
        valid_intents = [i.value for i in Intent]
        if intent not in valid_intents:
            logger.warning(f"Invalid intent '{intent}', defaulting to fallback")
            intent = Intent.FALLBACK.value

        logger.info(f"Classified intent: {intent}, entities: {entities}")

        return {
            "intent": intent,
            "entities": entities,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error in router node: {e}")
        return {
            "intent": Intent.FALLBACK.value,
            "entities": {},
            "error": str(e),
        }


def route_intent(state: AgentState) -> str:
    """Routing function for conditional edges.

    Determines which agent node to execute next based on classified intent.

    Args:
        state: Current agent state with intent field.

    Returns:
        Next node name as string.

    Example:
        >>> state = {"intent": "pnl_summary"}
        >>> next_node = route_intent(state)
        >>> print(next_node)  # "pnl_agent"
    """
    intent = state.get("intent", Intent.FALLBACK.value)

    # Map intents to agent nodes
    if intent in [Intent.PNL_SUMMARY.value, Intent.PNL_BREAKDOWN.value]:
        return "pnl_agent"
    elif intent in [Intent.PROPERTY_DETAILS.value, Intent.PROPERTY_COMPARE.value]:
        return "property_agent"
    elif intent in [Intent.TENANT_DETAILS.value, Intent.TENANT_RANKING.value]:
        return "tenant_agent"
    elif intent == Intent.GENERAL_KNOWLEDGE.value:
        return "general_agent"
    else:
        return "fallback"
