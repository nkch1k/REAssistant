"""Data loader module with singleton pattern for parquet file.

This module provides a singleton DataLoader class that loads and caches
the parquet file data for efficient access throughout the application.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATA_FILE

logger = logging.getLogger(__name__)


class DataLoader:
    """Singleton data loader for parquet file.

    This class ensures that the parquet file is loaded only once and
    cached for subsequent access throughout the application lifecycle.
    """

    _instance: Optional["DataLoader"] = None
    _df: Optional[pd.DataFrame] = None

    def __new__(cls) -> "DataLoader":
        """Create or return the singleton instance.

        Returns:
            The singleton DataLoader instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the data loader (only runs once)."""
        if self._df is None:
            self._load_data()

    def _load_data(self) -> None:
        """Load the parquet file into memory.

        Raises:
            FileNotFoundError: If the parquet file doesn't exist.
            Exception: If there's an error reading the file.
        """
        try:
            file_path = Path(DATA_FILE)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

            logger.info(f"Loading data from {DATA_FILE}")
            self._df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(self._df)} rows from {DATA_FILE}")

            # Validate required columns
            required_columns = {
                "entity_name",
                "property_name",
                "tenant_name",
                "ledger_type",
                "ledger_group",
                "ledger_category",
                "ledger_code",
                "ledger_description",
                "month",
                "quarter",
                "year",
                "profit",
            }

            missing_columns = required_columns - set(self._df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info("Data validation successful")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    @property
    def data(self) -> pd.DataFrame:
        """Get the loaded dataframe.

        Returns:
            The loaded pandas DataFrame.

        Raises:
            RuntimeError: If data hasn't been loaded successfully.
        """
        if self._df is None:
            raise RuntimeError("Data not loaded")
        return self._df

    def reload(self) -> None:
        """Reload the data from the parquet file.

        This method can be called to refresh the data if the file has changed.
        """
        logger.info("Reloading data")
        self._df = None
        self._load_data()


def get_dataframe() -> pd.DataFrame:
    """Get the singleton dataframe instance.

    Returns:
        The cached pandas DataFrame.

    Example:
        >>> df = get_dataframe()
        >>> total_profit = df['profit'].sum()
    """
    loader = DataLoader()
    return loader.data
