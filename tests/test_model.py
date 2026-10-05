"""
tests/test_model.py
--------------------
Tests for model training, predictions, and saved artefacts.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import joblib

from src.config             import (
    BEST_MODEL_PKL, PIPELINE_PKL, FEATURE_THRESHOLDS_PKL,
    TARGET_COL, ID_COL,
)
from src.feature_engineering import apply_charge_features
from src.data_loader        import load_churn_data
from src.data_cleaning      import clean_data
from src.feature_engineering import engineer_features
from src.preprocessing      import split_and_preprocess
from src.modeling           import train_all_models


@pytest.fixture(scope="module")
def pipeline_results():
    df     = load_churn_data()
    df     = clean_data(df, verbose=False)
    df     = engineer_features(df)
    splits = split_and_preprocess(df)
    X_train, X_test, y_train, y_test = splits[:4]
    models = train_all_models(X_train, y_train)
    return models, X_train, X_test, y_train, y_test


def test_minimum_four_models(pipeline_results):
    models, *_ = pipeline_results
    assert len(models) >= 4


def test_predictions_correct_length(pipeline_results):
    models, _, X_test, _, y_test = pipeline_results
    for name, model in models.items():
        assert len(model.predict(X_test)) == len(y_test), \
            f"{name}: wrong prediction length"


def test_predictions_binary(pipeline_results):
    models, _, X_test, _, _ = pipeline_results
    for name, model in models.items():
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1}), f"{name}: non-binary predictions"


def test_probabilities_in_range(pipeline_results):
    models, _, X_test, _, _ = pipeline_results
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all(), \
            f"{name}: probability out of [0,1]"


def test_saved_model_exists():
    assert BEST_MODEL_PKL.exists(), "best_model.pkl not found — run pipeline first"


def test_saved_pipeline_exists():
    assert PIPELINE_PKL.exists(), "preprocessing_pipeline.pkl not found"


def test_saved_model_produces_valid_output():
    if not BEST_MODEL_PKL.exists() or not PIPELINE_PKL.exists() or not FEATURE_THRESHOLDS_PKL.exists():
        pytest.skip("Artefacts not present")
    model   = joblib.load(BEST_MODEL_PKL)
    pipe    = joblib.load(PIPELINE_PKL)
    df      = load_churn_data()
    df      = clean_data(df, verbose=False)
    df      = engineer_features(df)
    thresholds = joblib.load(FEATURE_THRESHOLDS_PKL)
    df      = apply_charge_features(df, thresholds)
    X       = df.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in df.columns])
    X_proc  = pipe.transform(X)
    probs   = model.predict_proba(X_proc)[:, 1]
    assert len(probs) == len(df)
    assert (probs >= 0).all() and (probs <= 1).all()
