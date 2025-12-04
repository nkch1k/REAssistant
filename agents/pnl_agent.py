"""P&L agent for profit and loss queries with LLM-based response generation.

This module handles all P&L queries using LLM to interpret complex requests
and generate flexible responses.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import ERROR_NO_DATA, OPENAI_MODEL
from data.queries import get_pnl_breakdown, get_total_pnl
from state import AgentState

logger = logging.getLogger(__name__)


def pnl_agent_node(state: AgentState) -> dict[str, Any]:
    """P&L agent node with LLM-based query handling.

    This node handles P&L queries by:
    1. Retrieving relevant P&L data
    2. Using LLM to understand the query and generate response
    3. Supporting any variation of P&L-related questions

    Args:
        state: Current agent state with user_query and entities.

    Returns:
        State update dict with data and response fields.

    Example:
        >>> state = {"user_query": "What are my biggest expenses?"}
        >>> result = pnl_agent_node(state)
    """
    user_query = state["user_query"]
    entities = state["entities"]
    logger.info(f"P&L agent processing query: {user_query}")

    try:
        # Extract time period filters
        year = entities.get("year")
        quarter = entities.get("quarter")

        # Get both summary and breakdown data
        total_pnl = get_total_pnl(year=year, quarter=quarter)
        breakdown = get_pnl_breakdown(year=year)

        if total_pnl == 0 and not breakdown:
            return {
                "data": {},
                "response": ERROR_NO_DATA,
                "error": None,
            }

        # Create context for LLM
        period = quarter if quarter else (f"year {year}" if year else "all periods")

        # Format breakdown data
        context = f"""P&L Data for {period}:

Total P&L: ${total_pnl:,.2f}

Breakdown by Category:"""

        # Separate revenue and expenses
        revenue_items = []
        expense_items = []

        for category, amount in breakdown.items():
            category_name = category.replace('_', ' ').title()
            if amount > 0:
                revenue_items.append(f"- {category_name}: ${amount:,.2f}")
            else:
                expense_items.append(f"- {category_name}: ${abs(amount):,.2f}")

        if revenue_items:
            context += "\n\nRevenue:\n" + "\n".join(revenue_items)
            total_revenue = sum(amt for _, amt in [(c, breakdown[c]) for c in breakdown if breakdown[c] > 0])
            context += f"\nTotal Revenue: ${total_revenue:,.2f}"

        if expense_items:
            context += "\n\nExpenses:\n" + "\n".join(expense_items)
            total_expenses = sum(abs(amt) for _, amt in [(c, breakdown[c]) for c in breakdown if breakdown[c] < 0])
            context += f"\nTotal Expenses: ${total_expenses:,.2f}"

        # Create prompt for LLM
        prompt = f"""{context}

User Question: {user_query}

IMPORTANT FORMATTING RULES:
- Always format currency as $X,XXX.XX (with dollar sign and commas)
- NEVER use markdown ** inside currency values
- Use markdown ** only for emphasis on TEXT, not numbers
- Keep response brief and direct
- If asked about biggest/smallest expenses or revenue, identify and highlight them
- If asked for comparisons or trends, explain based on the data provided

Provide a clear, concise answer using ONLY the data above."""

        # Initialize LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )

        # Get LLM response
        response_message = llm.invoke([HumanMessage(content=prompt)])
        response = response_message.content

        logger.info("P&L agent completed successfully")
        return {
            "data": {
                "total_pnl": total_pnl,
                "breakdown": breakdown,
                "year": year,
                "quarter": quarter,
            },
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
