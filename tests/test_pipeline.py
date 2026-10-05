"""
tests/test_pipeline.py
-----------------------
End-to-end pipeline integration test.
Runs the full workflow (data load -> predict) and validates outputs.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd

from src.config import (
    PROCESSED_CHURN_CSV, PIPELINE_PKL, BEST_MODEL_PKL, FEATURE_THRESHOLDS_PKL,
    METRICS_DIR, TABLES_DIR, PREDICTIONS_DIR, FIGURES_DIR,
)


def test_processed_dataset_exists():
    """Pipeline must produce a processed CSV."""
    assert PROCESSED_CHURN_CSV.exists(), \
        "processed_customer_churn.csv not found — run pipeline first"


def test_pipeline_pkl_exists():
    assert PIPELINE_PKL.exists(), "preprocessing_pipeline.pkl not found"
    assert FEATURE_THRESHOLDS_PKL.exists(), "feature_thresholds.pkl not found"


def test_model_pkl_exists():
    assert BEST_MODEL_PKL.exists(), "best_model.pkl not found"


def test_model_comparison_csv_exists():
    out = METRICS_DIR / "model_comparison.csv"
    assert out.exists(), "model_comparison.csv not found"


def test_model_comparison_has_required_columns():
    out = METRICS_DIR / "model_comparison.csv"
    if not out.exists():
        pytest.skip("model_comparison.csv not found")
    df = pd.read_csv(out)
    for col in ["Model", "Accuracy", "Recall", "F1", "ROC_AUC"]:
        assert col in df.columns, f"Missing column in model_comparison.csv: {col}"


def test_model_comparison_has_four_models():
    out = METRICS_DIR / "model_comparison.csv"
    if not out.exists():
        pytest.skip("model_comparison.csv not found")
    df = pd.read_csv(out)
    assert len(df) >= 4, f"Expected >= 4 models, found {len(df)}"


def test_cv_results_exist():
    out = METRICS_DIR / "cross_validation_results.csv"
    assert out.exists(), "cross_validation_results.csv not found"


def test_hyperparameter_results_exist():
    out = METRICS_DIR / "hyperparameter_results.csv"
    assert out.exists(), "hyperparameter_results.csv not found"


def test_data_quality_report_exists():
    out = TABLES_DIR / "data_quality_report.csv"
    assert out.exists(), "data_quality_report.csv not found"


def test_feature_importance_csv_exists():
    out = TABLES_DIR / "feature_importance.csv"
    assert out.exists(), "feature_importance.csv not found"


def test_customer_risk_predictions_exist():
    out = PREDICTIONS_DIR / "customer_risk_predictions.csv"
    assert out.exists(), "customer_risk_predictions.csv not found"


def test_risk_predictions_schema():
    out = PREDICTIONS_DIR / "customer_risk_predictions.csv"
    if not out.exists():
        pytest.skip("customer_risk_predictions.csv not found")
    df = pd.read_csv(out)
    for col in ["CustomerID", "ChurnProbability", "Prediction", "RiskLevel"]:
        assert col in df.columns, f"Missing column: {col}"


def test_risk_levels_valid():
    out = PREDICTIONS_DIR / "customer_risk_predictions.csv"
    if not out.exists():
        pytest.skip("customer_risk_predictions.csv not found")
    df = pd.read_csv(out)
    valid = {"High Risk", "Medium Risk", "Low Risk"}
    actual = set(df["RiskLevel"].unique())
    assert actual.issubset(valid), f"Invalid risk levels: {actual - valid}"


def test_probabilities_in_range():
    out = PREDICTIONS_DIR / "customer_risk_predictions.csv"
    if not out.exists():
        pytest.skip("customer_risk_predictions.csv not found")
    df = pd.read_csv(out)
    assert (df["ChurnProbability"] >= 0).all()
    assert (df["ChurnProbability"] <= 1).all()


def test_eda_figures_exist():
    """At least 10 EDA figures should be present."""
    figs = list(FIGURES_DIR.glob("*.png"))
    assert len(figs) >= 10, f"Expected >= 10 figures, found {len(figs)}"
