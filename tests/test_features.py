"""
tests/test_features.py
-----------------------
Tests for feature engineering on the actual 9-column dataset.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.data_loader       import load_churn_data
from src.data_cleaning     import clean_data
from src.feature_engineering import engineer_features

REQUIRED_FEATURES = [
    "AverageMonthlySpend",
    "ChargesPerTenure",
    "EstimatedCLV",
    "TenureGroup",
    "ContractRisk",
    "PaymentRisk",
    "HighChargeFlag",
    "CustomerValueSegment",
]


@pytest.fixture(scope="module")
def eng_df():
    df = load_churn_data()
    df = clean_data(df, verbose=False)
    return engineer_features(df, fit_charge_thresholds=True)


@pytest.mark.parametrize("feature", REQUIRED_FEATURES)
def test_feature_exists(eng_df, feature):
    assert feature in eng_df.columns, f"Missing feature: {feature}"


def test_estimated_clv_non_negative(eng_df):
    assert (eng_df["EstimatedCLV"] >= 0).all()


def test_average_monthly_spend_non_negative(eng_df):
    assert (eng_df["AverageMonthlySpend"] >= 0).all()


def test_contract_risk_binary(eng_df):
    assert set(eng_df["ContractRisk"].unique()).issubset({0, 1})


def test_payment_risk_binary(eng_df):
    assert set(eng_df["PaymentRisk"].unique()).issubset({0, 1})


def test_high_charge_flag_binary(eng_df):
    assert set(eng_df["HighChargeFlag"].unique()).issubset({0, 1})


def test_tenure_group_valid(eng_df):
    valid = {"New", "Growing", "Established", "Loyal"}
    actual = set(eng_df["TenureGroup"].unique())
    assert actual.issubset(valid), f"Unexpected TenureGroup values: {actual - valid}"


def test_no_inf_values(eng_df):
    for col in ["AverageMonthlySpend", "ChargesPerTenure", "EstimatedCLV"]:
        assert not np.isinf(eng_df[col]).any(), f"Inf values in {col}"
