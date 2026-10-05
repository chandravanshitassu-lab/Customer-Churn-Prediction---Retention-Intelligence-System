"""
data_loader.py
--------------
Loads the actual provided datasets from data/raw/.
Does NOT generate synthetic data.
"""

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    RAW_CHURN_CSV, RAW_SALES_CSV, RAW_HOUSE_CSV,
    EXPECTED_COLUMNS, TARGET_COL, ID_COL,
)


def load_churn_data() -> pd.DataFrame:
    """
    Load the primary customer churn dataset.

    Returns
    -------
    pd.DataFrame
        Raw churn dataset (500 rows x 9 columns).

    Raises
    ------
    FileNotFoundError
        If customer_churn.csv is not found in data/raw/.
    """
    if not RAW_CHURN_CSV.exists():
        raise FileNotFoundError(
            f"Primary dataset not found: {RAW_CHURN_CSV}\n"
            "Please ensure customer_churn.csv is in data/raw/."
        )
    df = pd.read_csv(RAW_CHURN_CSV)
    print(f"  [OK] Loaded churn dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def load_sales_data() -> pd.DataFrame:
    """Load the supporting sales dataset."""
    if not RAW_SALES_CSV.exists():
        print(f"  [WARN] sales_data.csv not found at {RAW_SALES_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(RAW_SALES_CSV)
    print(f"  [OK] Loaded sales dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def load_house_data() -> pd.DataFrame:
    """Load the supporting house prices dataset."""
    if not RAW_HOUSE_CSV.exists():
        print(f"  [WARN] house_prices.csv not found at {RAW_HOUSE_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(RAW_HOUSE_CSV)
    print(f"  [OK] Loaded house prices dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    df = load_churn_data()
    print(df.head())
