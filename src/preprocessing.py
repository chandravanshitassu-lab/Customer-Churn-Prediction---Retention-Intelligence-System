"""
preprocessing.py
----------------
Builds and fits a sklearn ColumnTransformer pipeline for the 9-column
customer churn dataset. Saves the fitted pipeline to models/.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    TARGET_COL, ID_COL, PIPELINE_PKL, FEATURE_THRESHOLDS_PKL,
    RANDOM_STATE, TEST_SIZE,
)
from src.feature_engineering import (
    CHARGE_FEATURE_COLS,
    learn_charge_thresholds,
    apply_charge_features,
)

# ---------------------------------------------------------------------------
# Feature column lists  (including engineered features)
# ---------------------------------------------------------------------------
_BASE_NUM  = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
_ENG_NUM   = ["AverageMonthlySpend", "ChargesPerTenure", "EstimatedCLV"]
_BASE_CAT  = ["Contract", "PaymentMethod", "PaperlessBilling"]
_ENG_CAT   = ["TenureGroup", "CustomerValueSegment"]
_BINARY    = ["ContractRisk", "PaymentRisk", "HighChargeFlag"]   # treated as numeric


def build_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Build the ColumnTransformer preprocessing pipeline."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def split_and_preprocess(df: pd.DataFrame):
    """
    Split the dataset and fit-transform the preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned + feature-engineered dataset.

    Returns
    -------
    tuple
        X_train_proc, X_test_proc, y_train, y_test,
        preprocessor, feature_names, X_train_raw, X_test_raw
    """
    drop_cols = [c for c in [ID_COL] if c in df.columns]
    df_model  = df.drop(columns=drop_cols)
    # Never keep charge features computed on the full dataset
    leak_cols = [c for c in CHARGE_FEATURE_COLS if c in df_model.columns]
    if leak_cols:
        df_model = df_model.drop(columns=leak_cols)

    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    charge_thresholds = learn_charge_thresholds(X_train["MonthlyCharges"])
    X_train = apply_charge_features(X_train, charge_thresholds)
    X_test  = apply_charge_features(X_test, charge_thresholds)

    FEATURE_THRESHOLDS_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(charge_thresholds, FEATURE_THRESHOLDS_PKL)
    print(
        "  Charge cuts (train only): "
        f"q33={charge_thresholds['q33']:.2f}, "
        f"q66={charge_thresholds['q66']:.2f}, "
        f"q75={charge_thresholds['q75']:.2f}"
    )

    num_cols = [c for c in (_BASE_NUM + _ENG_NUM + _BINARY) if c in X_train.columns]
    cat_cols = [c for c in (_BASE_CAT + _ENG_CAT) if c in X_train.columns]

    print("\n  --- Train / Test Split ---")
    print(f"  Training set : {len(X_train)} rows ({1 - TEST_SIZE:.0%})")
    print(f"  Test set     : {len(X_test)} rows ({TEST_SIZE:.0%})")
    print(f"  Churn (train): {y_train.sum()} positive ({y_train.mean()*100:.1f}%)")
    print(f"  Churn (test) : {y_test.sum()} positive ({y_test.mean()*100:.1f}%)")
    print(f"  Stratified   : Yes | Random state: {RANDOM_STATE}")

    preprocessor   = build_pipeline(num_cols, cat_cols)
    X_train_proc   = preprocessor.fit_transform(X_train)
    X_test_proc    = preprocessor.transform(X_test)

    # Feature names after OHE
    ohe           = preprocessor.named_transformers_["cat"]["encoder"]
    cat_feat_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names  = num_cols + cat_feat_names

    # Save fitted pipeline
    PIPELINE_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PIPELINE_PKL)
    print(f"  [OK] Pipeline saved -> {PIPELINE_PKL.name}")

    return (
        X_train_proc, X_test_proc,
        y_train, y_test,
        preprocessor,
        feature_names,
        X_train, X_test,
    )
