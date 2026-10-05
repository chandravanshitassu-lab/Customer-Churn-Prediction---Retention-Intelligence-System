"""
data_validation.py
------------------
Programmatic data quality checks for the customer churn dataset.
Reports actual findings -- does NOT fabricate problems.
Saves: outputs/tables/data_quality_report.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    RAW_CHURN_CSV, TARGET_COL, ID_COL,
    NUMERICAL_COLS, CATEGORICAL_COLS,
    EXPECTED_COLUMNS, VALID_CATEGORIES,
    TABLES_DIR, ROOT_DIR,
)


def run_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all data quality checks and save a report CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Raw churn dataset.

    Returns
    -------
    pd.DataFrame
        Quality report: [Check, Result, Details, Status]
    """
    rows = []

    def _add(check, result, details, status):
        icon = "[OK]" if status == "PASS" else ("[!]" if status == "WARN" else "[X]")
        print(f"    {icon}  [{status}] {check}: {result}")
        rows.append({"Check": check, "Result": str(result),
                     "Details": details, "Status": status})

    print("\n  --- Data Validation ---")

    # 1. File exists
    try:
        file_details = str(RAW_CHURN_CSV.relative_to(ROOT_DIR))
    except ValueError:
        file_details = RAW_CHURN_CSV.name
    _add("File Exists", RAW_CHURN_CSV.name,
         file_details,
         "PASS" if RAW_CHURN_CSV.exists() else "FAIL")

    # 2. Row count
    n = len(df)
    _add("Row Count", n, f"Expected ~500", "PASS" if n >= 100 else "WARN")

    # 3. Column count
    _add("Column Count", len(df.columns),
         f"Columns: {list(df.columns)}",
         "PASS" if len(df.columns) >= 5 else "FAIL")

    # 4. Expected columns present
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    _add("Expected Columns",
         "All present" if not missing_cols else str(missing_cols),
         f"{len(missing_cols)} missing",
         "PASS" if not missing_cols else "FAIL")

    # 5. Target column exists
    _add("Target Column Exists", TARGET_COL,
         f"'{TARGET_COL}' in columns",
         "PASS" if TARGET_COL in df.columns else "FAIL")

    # 6. Target values valid
    if TARGET_COL in df.columns:
        unique_t = set(df[TARGET_COL].dropna().unique())
        _add("Target Values (0/1)", str(unique_t),
             "Must be {0, 1}",
             "PASS" if unique_t.issubset({0, 1}) else "FAIL")

        churn_rate = df[TARGET_COL].mean() * 100
        status = "PASS" if 1 <= churn_rate <= 80 else "WARN"
        _add("Churn Rate (%)", f"{churn_rate:.1f}%",
             "Actual churn proportion", status)

    # 7. Missing values
    mv = df.isnull().sum()
    total_mv = int(mv.sum())
    missing_detail = (
        ", ".join(f"{c}:{mv[c]}" for c in df.columns if mv[c] > 0) or "None"
    )
    _add("Missing Values", f"{total_mv} total",
         missing_detail,
         "PASS" if total_mv == 0 else "WARN")

    # 8. Duplicate rows
    n_dup = int(df.duplicated().sum())
    _add("Duplicate Rows", n_dup,
         f"{n_dup} exact duplicates",
         "PASS" if n_dup == 0 else "WARN")

    # 9. Duplicate CustomerIDs
    if ID_COL in df.columns:
        n_dup_id = int(df[ID_COL].duplicated().sum())
        _add("Duplicate CustomerIDs", n_dup_id,
             f"{n_dup_id} duplicate IDs",
             "PASS" if n_dup_id == 0 else "FAIL")

    # 10. Numerical ranges
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        n_neg = int((df[col] < 0).sum())
        _add(f"Non-Negative: {col}",
             f"min={col_min}, max={col_max}",
             f"{n_neg} negative values",
             "PASS" if n_neg == 0 else "FAIL")

    # 11. SeniorCitizen is binary
    if "SeniorCitizen" in df.columns:
        sc_vals = set(df["SeniorCitizen"].dropna().unique())
        _add("SeniorCitizen Binary", str(sc_vals),
             "Must be {0, 1}",
             "PASS" if sc_vals.issubset({0, 1}) else "FAIL")

    # 12. Categorical validity
    for col, valid_set in VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().str.strip().unique())
        unexpected = actual - valid_set
        _add(f"Valid Categories: {col}",
             "OK" if not unexpected else str(unexpected),
             f"Valid: {valid_set}",
             "PASS" if not unexpected else "WARN")

    # 13. Numerical data types
    for col in NUMERICAL_COLS:
        if col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            _add(f"Numeric dtype: {col}", str(df[col].dtype),
                 "Expected numeric",
                 "PASS" if is_num else "WARN")

    # 14. Outlier detection (IQR method) - informational only
    for col in ["Tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 3 * iqr)) | (df[col] > (q3 + 3 * iqr))).sum()
        _add(f"Extreme Outliers (3xIQR): {col}",
             f"{outliers} found",
             f"IQR={iqr:.1f}, Q1={q1:.1f}, Q3={q3:.1f}",
             "PASS" if outliers == 0 else "WARN")

    report_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "data_quality_report.csv"
    report_df.to_csv(out_path, index=False)

    pass_n = (report_df["Status"] == "PASS").sum()
    warn_n = (report_df["Status"] == "WARN").sum()
    fail_n = (report_df["Status"] == "FAIL").sum()
    print(f"\n  Summary: {pass_n} PASS | {warn_n} WARN | {fail_n} FAIL")
    print(f"  [OK] Quality report saved -> {out_path.name}")
    return report_df
