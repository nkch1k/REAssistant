"""General knowledge agent for portfolio-level queries.

This module handles general knowledge questions about the portfolio using
LLM-based response generation with data retrieval.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL
from data.queries import get_portfolio_stats
from state import AgentState

logger = logging.getLogger(__name__)


def general_agent_node(state: AgentState) -> dict[str, Any]:
    """General knowledge agent node for portfolio queries.

    This node handles general questions about the portfolio by:
    1. Retrieving comprehensive portfolio statistics
    2. Using LLM to generate natural language response
    3. Providing context-aware answers

    Args:
        state: Current agent state with user_query.

    Returns:
        State update dict with data and response fields.

    Example:
        >>> state = {"user_query": "How many tenants do I have?"}
        >>> result = general_agent_node(state)
        >>> print(result["response"])  # "You have 18 tenants in total."
    """
    user_query = state["user_query"]
    logger.info(f"General agent processing query: {user_query}")

    try:
        # Get portfolio statistics
        stats = get_portfolio_stats()

        # Format property and tenant lists for LLM
        properties_str = ", ".join(stats["properties"])
        years_str = ", ".join(stats["years_covered"])

        # Create context for LLM
        context = f"""Portfolio Information:
- Number of Properties: {stats['property_count']}
- Properties: {properties_str}
- Number of Tenants: {stats['tenant_count']}
- Total Revenue: ${stats['total_revenue']:,.2f}
- Total Expenses: ${stats['total_expenses']:,.2f}
- Net P&L: ${stats['net_pnl']:,.2f}
- Years with Data: {years_str}"""

        # Create prompt for LLM
        prompt = f"""{context}

User Question: {user_query}

CRITICAL FORMATTING RULES - DO NOT DEVIATE:
1. Currency values: Write as plain text with $ sign, NO markdown formatting
   ✓ CORRECT: "Total revenue of $5,000,000.00"
   ✗ WRONG: "Total revenue of $5,000,000.00**" or "Total revenue of **$5,000,000.00**"

2. Numbers: NEVER put ** before or after any number
   ✓ CORRECT: "You have 18 tenants across 6 properties"
   ✗ WRONG: "You have **18** tenants across **6** properties"

3. Text emphasis: Use ** only around WORDS, never numbers
   ✓ CORRECT: "Your portfolio has **strong** performance"
   ✗ WRONG: "Your portfolio has **18 tenants**"

Keep response brief and direct.

Provide a clear, concise answer using ONLY the data above."""

        # Initialize LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )

        # Get LLM response
        response_message = llm.invoke([HumanMessage(content=prompt)])
        response = response_message.content

        logger.info("General agent completed successfully")
        return {
            "data": stats,
            "response": response,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error in general agent: {e}")
        return {
            "data": {},
            "response": f"Error processing general query: {str(e)}",
            "error": str(e),
        }
