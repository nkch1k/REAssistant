"""Tenant agent for tenant-related queries with LLM-based response generation.

This module handles all tenant queries using LLM to interpret complex requests
and generate flexible responses.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import ERROR_TENANT_NOT_FOUND, OPENAI_MODEL
from data.queries import (
    fuzzy_match_tenant,
    get_all_tenants_with_revenue,
    get_tenant_revenue,
)
from state import AgentState

logger = logging.getLogger(__name__)


def tenant_agent_node(state: AgentState) -> dict[str, Any]:
    """Tenant agent node with LLM-based query handling.

    This node handles tenant queries by:
    1. Retrieving relevant tenant data
    2. Using LLM to understand the query and generate response
    3. Supporting any variation of tenant-related questions

    Args:
        state: Current agent state with user_query and entities.

    Returns:
        State update dict with data and response fields.

    Example:
        >>> state = {"user_query": "Show me top 5 tenants"}
        >>> result = tenant_agent_node(state)
    """
    user_query = state["user_query"]
    entities = state["entities"]
    logger.info(f"Tenant agent processing query: {user_query}")

    try:
        # Check if querying a specific tenant
        tenant_name = entities.get("tenant_name")

        if tenant_name:
            # Specific tenant query
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

            # Create context for LLM
            context = f"""Tenant Data:
- Tenant: {matched_name}
- Total Revenue: ${revenue:,.2f}"""

            if year:
                context += f"\n- Year: {year}"

            data = {
                "tenant_name": matched_name,
                "total_revenue": revenue,
                "year": year,
            }

        else:
            # General tenant query (rankings, comparisons, etc.)
            all_tenants = get_all_tenants_with_revenue()

            if not all_tenants:
                return {
                    "data": {},
                    "response": "No tenant data available.",
                    "error": None,
                }

            # Format all tenants for LLM
            tenants_list = []
            for tenant in all_tenants:
                tenants_list.append(
                    f"{tenant['rank']}. {tenant['tenant_name']}: "
                    f"Revenue=${tenant['total_revenue']:,.2f}"
                )

            context = "All Tenants (ranked by revenue):\n" + "\n".join(tenants_list)
            data = {"all_tenants": all_tenants}

        # Create prompt for LLM
        prompt = f"""{context}

User Question: {user_query}

CRITICAL FORMATTING RULES - DO NOT DEVIATE:
1. Currency values: Write as plain text with $ sign, NO markdown formatting
   ✓ CORRECT: "Revenue of $250,000.00"
   ✗ WRONG: "Revenue of $250,000.00**" or "Revenue of **$250,000.00**"

2. Numbers: NEVER put ** before or after any number
   ✓ CORRECT: "Tenant 8 with revenue of $250,000.00"
   ✗ WRONG: "Tenant 8 with revenue of $250,000.00**"

3. Text emphasis: Use ** only around WORDS, never numbers
   ✓ CORRECT: "Your **best** tenant is Tenant 8"
   ✗ WRONG: "Your best tenant is **Tenant 8 with $250,000.00**"

Keep response brief and direct. If ranking tenants, show comparison clearly.

Provide a clear, concise answer using ONLY the data above."""

        # Initialize LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )

        # Get LLM response
        response_message = llm.invoke([HumanMessage(content=prompt)])
        response = response_message.content

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
