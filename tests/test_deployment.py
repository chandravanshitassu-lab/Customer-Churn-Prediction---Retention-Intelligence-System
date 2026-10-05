"""
tests/test_deployment.py
-------------------------
Tests for the deployment prediction utility (actual 9-column schema).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from deployment.prediction import predict_single_customer, validate_inputs

VALID_CUSTOMER = {
    "Tenure":            6,
    "MonthlyCharges":    85,
    "TotalCharges":      510,
    "Contract":          "Month-to-month",
    "PaymentMethod":     "Electronic Check",
    "PaperlessBilling":  "Yes",
    "SeniorCitizen":     1,
}


@pytest.fixture(scope="module")
def prediction():
    from src.config import BEST_MODEL_PKL, PIPELINE_PKL
    if not BEST_MODEL_PKL.exists() or not PIPELINE_PKL.exists():
        pytest.skip("Saved artefacts not present — run pipeline first")
    return predict_single_customer(VALID_CUSTOMER)


def test_prediction_runs(prediction):
    assert prediction is not None


def test_prediction_has_keys(prediction):
    for key in ["prediction", "probability", "risk_level", "recommendation"]:
        assert key in prediction, f"Missing key: {key}"


def test_prediction_binary(prediction):
    assert prediction["prediction"] in {0, 1}


def test_probability_range(prediction):
    p = prediction["probability"]
    assert 0.0 <= p <= 1.0, f"Probability {p} out of [0,1]"


def test_risk_level_valid(prediction):
    assert prediction["risk_level"] in {"High Risk", "Medium Risk", "Low Risk"}


def test_recommendation_not_empty(prediction):
    assert len(prediction["recommendation"]) > 10


# --- Input validation tests ---

def test_valid_inputs_pass():
    errors = validate_inputs(VALID_CUSTOMER)
    assert errors == []


def test_missing_tenure_fails():
    bad = {**VALID_CUSTOMER, "Tenure": None}
    assert any("Tenure" in e for e in validate_inputs(bad))


def test_negative_monthly_fails():
    bad = {**VALID_CUSTOMER, "MonthlyCharges": -5}
    assert any("MonthlyCharges" in e for e in validate_inputs(bad))


def test_non_numeric_tenure_fails():
    bad = {**VALID_CUSTOMER, "Tenure": "abc"}
    assert any("Tenure" in e for e in validate_inputs(bad))
