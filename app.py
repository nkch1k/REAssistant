"""Streamlit UI for Real Estate Multi-Agent System.

This module provides a simple chat interface for interacting with
the multi-agent real estate assistant.
"""

import logging
from datetime import datetime

import streamlit as st

from config import (
    OPENAI_API_KEY,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_PAGE_TITLE,
)
from graph import run_query

# Configure logging with DEBUG level for troubleshooting
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Set all related loggers to DEBUG
logging.getLogger("graph").setLevel(logging.DEBUG)
logging.getLogger("agents").setLevel(logging.DEBUG)
logging.getLogger("data").setLevel(logging.DEBUG)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables.

    Creates persistent storage for chat history and other state.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized session state")

    if "processing" not in st.session_state:
        st.session_state.processing = False


def display_chat_history() -> None:
    """Display all messages from chat history.

    Renders past messages in a chat-like format.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_query(query: str) -> str:
    """Process user query through the multi-agent system.

    Args:
        query: User's question or request.

    Returns:
        Response string from the agent system.
    """
    try:
        logger.info(f"=" * 60)
        logger.info(f"PROCESSING QUERY: {query}")
        logger.info(f"=" * 60)

        result = run_query(query)

        logger.info(f"Result keys: {result.keys()}")
        logger.info(f"Intent: {result.get('intent', 'N/A')}")
        logger.info(f"Error: {result.get('error', 'N/A')}")
        logger.info(f"Response length: {len(result.get('response', ''))}")

        response = result.get("response", "")

        if not response:
            logger.warning("Empty response received from run_query")
            response = "I received your query but couldn't generate a response. Please try again."

        logger.info(f"Returning response: {response[:100]}...")
        return response

    except Exception as e:
        logger.error(f"Exception in process_query: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"An error occurred while processing your query: {str(e)}"


def main() -> None:
    """Main Streamlit application entry point.

    Sets up the UI and handles user interactions.
    """
    # Page configuration
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout=STREAMLIT_LAYOUT,
    )

    # Title and description
    st.title(f"{STREAMLIT_PAGE_ICON} {STREAMLIT_PAGE_TITLE}")
    st.markdown(
        """
        Ask me about:
        - **P&L summaries** and breakdowns
        - **Property performance** and comparisons
        - **Tenant revenue** and rankings
        """
    )

    # Check API key
    if not OPENAI_API_KEY:
        st.error(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
        st.stop()

    # Initialize session state
    initialize_session_state()

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This is a multi-agent system for real estate asset management.

            **Features:**
            - Intent classification
            - Entity extraction
            - Fuzzy matching
            - Financial analysis

            **Powered by:**
            - LangGraph
            - OpenAI GPT-4o-mini
            - Streamlit
            """
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Display chat history
    display_chat_history()

    # Chat input
    if query := st.chat_input("Ask a question about your real estate portfolio..."):
        logger.info(f"Received chat input: {query}")

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Process query and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.session_state.processing = True
                response = process_query(query)
                st.markdown(response)
                st.session_state.processing = False

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
