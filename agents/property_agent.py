"""Property agent for property-related queries with LLM-based response generation.

This module handles all property queries using LLM to interpret complex requests
and generate flexible responses.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import ERROR_PROPERTY_NOT_FOUND, OPENAI_MODEL
from data.queries import (
    fuzzy_match_property,
    get_all_properties_with_pnl,
    get_property_pnl,
    get_property_summary,
)
from state import AgentState

logger = logging.getLogger(__name__)


def property_agent_node(state: AgentState) -> dict[str, Any]:
    """Property agent node with LLM-based query handling.

    This node handles property queries by:
    1. Retrieving relevant property data
    2. Using LLM to understand the query and generate response
    3. Supporting any variation of property-related questions

    Args:
        state: Current agent state with user_query and entities.

    Returns:
        State update dict with data and response fields.

    Example:
        >>> state = {"user_query": "Show me best and worst properties"}
        >>> result = property_agent_node(state)
    """
    user_query = state["user_query"]
    entities = state["entities"]
    logger.info(f"Property agent processing query: {user_query}")

    try:
        # Check if querying a specific property
        property_name = entities.get("property_name")

        if property_name:
            # Specific property query
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
                property_data = get_property_pnl(matched_name, year=year)
            else:
                property_data = get_property_summary(matched_name)

            # Create context for LLM
            context = f"""Property Data:
- Property: {property_data['property_name']}
- Total P&L: ${property_data['total_pnl']:,.2f}
- Total Revenue: ${property_data['total_revenue']:,.2f}
- Total Expenses: ${abs(property_data['total_expenses']):,.2f}
- Tenant Count: {property_data['tenant_count']}"""

            if year:
                context += f"\n- Year: {year}"

            data = property_data

        else:
            # General property query (rankings, comparisons, etc.)
            all_properties = get_all_properties_with_pnl()

            if not all_properties:
                return {
                    "data": {},
                    "response": "No property data available.",
                    "error": None,
                }

            # Format all properties for LLM
            properties_list = []
            for idx, prop in enumerate(all_properties, 1):
                properties_list.append(
                    f"{idx}. {prop['property_name']}: "
                    f"P&L=${prop['total_pnl']:,.2f}, "
                    f"Revenue=${prop['total_revenue']:,.2f}, "
                    f"Expenses=${abs(prop['total_expenses']):,.2f}, "
                    f"Tenants={prop['tenant_count']}"
                )

            context = "All Properties (ranked by P&L):\n" + "\n".join(properties_list)
            data = {"all_properties": all_properties}

        # Create prompt for LLM
        prompt = f"""CRITICAL FORMATTING RULES - EVERY NUMBER NEEDS $:
1. ALL currency values MUST have $ sign - NO EXCEPTIONS
   ✓ CORRECT: "Best is Building 120 with $850,567.42, worst is Building 17 with $352,566.81"
   ✗ WRONG: "Best is Building 120 with 850,567.42, worst is Building 17 with 352,566.81" (missing $)
   ✗ WRONG: "Best is Building 120 with $850,567.42, worst is Building 17 with 352,566.81" (second missing $)

2. NEVER put ** around or after numbers
   ✓ CORRECT: "Your **best** property is Building 120"
   ✗ WRONG: "Building 120 with $850,567.42**"

{context}

User Question: {user_query}

Answer using ONLY the data above. Remember: EVERY currency number needs $ sign."""

        # Initialize LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )

        # Get LLM response
        response_message = llm.invoke([HumanMessage(content=prompt)])
        response = response_message.content

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
