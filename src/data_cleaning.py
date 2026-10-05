"""
data_cleaning.py  (called from run_project.py)
-----------------
Reproducible cleaning for the 9-column customer churn dataset.
All transformations are logged explicitly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_CHURN_CSV, CATEGORICAL_COLS


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the raw churn DataFrame.

    Steps
    -----
    1. Strip whitespace from column names.
    2. Strip leading/trailing whitespace from string values.
    3. Standardise categorical values (consistent casing).
    4. Convert numerical columns to proper dtypes.
    5. Remove exact duplicate rows.
    6. Clip numerical columns to valid ranges.
    7. Ensure Churn and SeniorCitizen are integer.

    Parameters
    ----------
    df : pd.DataFrame
    verbose : bool

    Returns
    -------
    pd.DataFrame  (cleaned copy)
    """
    df = df.copy()

    def _log(step, detail=""):
        if verbose:
            print(f"  [{step}] {detail}")

    if verbose:
        print("\n  --- Data Cleaning ---")

    # 1. Strip column name whitespace
    df.columns = [c.strip() for c in df.columns]
    _log("Column names", "Stripped whitespace")

    # 2. Strip string values
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "": np.nan, "None": np.nan})
    _log("String values", f"Stripped in {len(str_cols)} text columns")

    # 3. Standardise categorical values (already consistent but safe to check)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            before = df[col].nunique()
            # No case normalisation needed — values already consistent
            after = df[col].nunique()
            _log(f"Categorical: {col}", f"{before} -> {after} unique values")

    # 4. Convert numerical cols to numeric
    for col in ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if col not in df.columns:
            continue
        before_null = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_null = df[col].isnull().sum()
        new_nulls = after_null - before_null
        _log(f"Numeric: {col}",
             f"Coerced to numeric; {new_nulls} new NaN(s)")

    # 5. Handle missing values (impute if any arise)
    num_cols = ["Tenure", "MonthlyCharges", "TotalCharges"]
    for col in num_cols:
        n_miss = df[col].isnull().sum() if col in df.columns else 0
        if n_miss > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            _log(f"Impute {col}", f"Filled {n_miss} NaN(s) with median {med:.2f}")

    cat_null_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_null_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)
            _log(f"Impute {col}", f"Filled {n_miss} NaN(s) with mode '{mode_val}'")

    # 6. Remove duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup:
        df = df.drop_duplicates()
        _log("Duplicates", f"Removed {n_dup} duplicate rows")
    else:
        _log("Duplicates", "None found")

    # 7. Clip numerical to valid ranges
    clips = {"Tenure": (0, 999), "MonthlyCharges": (0, 9999),
             "TotalCharges": (0, 999999)}
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            n_clip = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            if n_clip:
                _log(f"Clip {col}", f"Clipped {n_clip} values to [{lo}, {hi}]")

    # 8. Ensure Churn and SeniorCitizen are int
    for col in ["Churn", "SeniorCitizen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    _log("Dtypes", "Churn and SeniorCitizen confirmed as int")

    # Save cleaned file
    PROCESSED_CHURN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CHURN_CSV, index=False)
    _log("Saved", f"Cleaned dataset -> {PROCESSED_CHURN_CSV.name} ({len(df)} rows)")

    return df
