"""Configuration and constants for the Real Estate Multi-Agent System.

This module contains all configuration settings, constants, and enums used
throughout the application.
"""

import os
from enum import Enum
from typing import Final

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# API Configuration
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: Final[str] = "gpt-4o-mini"


# Data Configuration
DATA_FILE: Final[str] = "cortex.parquet"
FUZZY_MATCH_THRESHOLD: Final[int] = 80


# Intent Types
class Intent(str, Enum):
    """User intent classifications."""

    PNL_SUMMARY = "pnl_summary"
    PNL_BREAKDOWN = "pnl_breakdown"
    PROPERTY_DETAILS = "property_details"
    TENANT_DETAILS = "tenant_details"
    TENANT_RANKING = "tenant_ranking"
    GENERAL_KNOWLEDGE = "general_knowledge"
    FALLBACK = "fallback"


# Ledger Types
class LedgerType(str, Enum):
    """Ledger type classifications."""

    EXPENSES = "expenses"
    REVENUE = "revenue"


# Entity Configuration
ENTITY_NAME: Final[str] = "PropCo"


# Property Names (from dataset)
PROPERTIES: Final[list[str]] = [
    "Building 120",
    "Building 140",
    "Building 160",
    "Building 17",
    "Building 180",
]


# Date Format
MONTH_FORMAT: Final[str] = "%Y-M%m"
QUARTER_FORMAT: Final[str] = "%Y-Q%q"


# UI Configuration
STREAMLIT_PAGE_TITLE: Final[str] = "Real Estate Assistant"
STREAMLIT_PAGE_ICON: Final[str] = "🏢"
STREAMLIT_LAYOUT: Final[str] = "centered"


# Error Messages
ERROR_PROPERTY_NOT_FOUND: Final[str] = "Property not found in the dataset."
ERROR_TENANT_NOT_FOUND: Final[str] = "Tenant not found in the dataset."
ERROR_NO_DATA: Final[str] = "No data available for the specified criteria."
ERROR_GENERAL: Final[str] = "An error occurred while processing your request."
