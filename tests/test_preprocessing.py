"""
tests/test_preprocessing.py
----------------------------
Tests for the preprocessing pipeline.
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
from src.preprocessing     import split_and_preprocess
from src.config            import PIPELINE_PKL


@pytest.fixture(scope="module")
def splits():
    df = load_churn_data()
    df = clean_data(df, verbose=False)
    df = engineer_features(df)
    return split_and_preprocess(df)


def test_split_returns_8_tuple(splits):
    assert len(splits) == 8


def test_train_is_larger_than_test(splits):
    X_train, X_test = splits[0], splits[1]
    assert X_train.shape[0] > X_test.shape[0]


def test_feature_count_consistent(splits):
    X_train, X_test = splits[0], splits[1]
    assert X_train.shape[1] == X_test.shape[1]


def test_no_nan_in_train(splits):
    X_train = splits[0]
    assert not np.isnan(X_train).any(), "NaN found in X_train"


def test_no_nan_in_test(splits):
    X_test = splits[1]
    assert not np.isnan(X_test).any(), "NaN found in X_test"


def test_labels_binary(splits):
    y_train, y_test = splits[2], splits[3]
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_pipeline_saved(splits):
    assert PIPELINE_PKL.exists(), "Pipeline PKL was not saved"


def test_stratification_preserved(splits):
    """Train and test should have similar churn rates."""
    y_train, y_test = splits[2], splits[3]
    assert abs(y_train.mean() - y_test.mean()) < 0.10, \
        "Churn rate differs too much between train and test"
