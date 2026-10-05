"""
tests/test_data.py
------------------
Tests for dataset loading and integrity (actual 9-column dataset).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.config import RAW_CHURN_CSV, TARGET_COL, ID_COL, EXPECTED_COLUMNS
from src.data_loader import load_churn_data


@pytest.fixture(scope="module")
def raw_df():
    return load_churn_data()


def test_dataset_file_exists():
    assert RAW_CHURN_CSV.exists(), f"Dataset not found: {RAW_CHURN_CSV}"


def test_dataset_loads(raw_df):
    assert isinstance(raw_df, pd.DataFrame)


def test_dataset_not_empty(raw_df):
    assert raw_df.shape[0] > 0
    assert raw_df.shape[1] > 0


def test_expected_columns(raw_df):
    for col in EXPECTED_COLUMNS:
        assert col in raw_df.columns, f"Missing column: {col}"


def test_target_column_exists(raw_df):
    assert TARGET_COL in raw_df.columns


def test_target_values_binary(raw_df):
    vals = set(raw_df[TARGET_COL].dropna().unique())
    assert vals.issubset({0, 1}), f"Non-binary target values: {vals}"


def test_no_missing_values(raw_df):
    total_missing = raw_df.isnull().sum().sum()
    assert total_missing == 0, f"{total_missing} missing values found"


def test_row_count(raw_df):
    assert len(raw_df) >= 100, f"Too few rows: {len(raw_df)}"


def test_customer_id_column(raw_df):
    assert ID_COL in raw_df.columns
    assert raw_df[ID_COL].duplicated().sum() == 0, "Duplicate CustomerIDs found"


def test_numerical_non_negative(raw_df):
    for col in ["Tenure", "MonthlyCharges", "TotalCharges"]:
        assert (raw_df[col] >= 0).all(), f"Negative values in {col}"
