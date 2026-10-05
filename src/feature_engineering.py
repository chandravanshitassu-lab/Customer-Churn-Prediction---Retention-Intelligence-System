"""
feature_engineering.py
-----------------------
Creates business-oriented engineered features from the actual 9-column
customer churn dataset.

Deterministic features use only the current row.
Charge-threshold features (HighChargeFlag, CustomerValueSegment) must be
learned from TRAINING data only, then applied with the saved cuts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURE_DOCS = {
    "AverageMonthlySpend": {
        "formula": "TotalCharges / max(Tenure, 1)",
        "rationale": "Smoothed monthly spend; corrects for varying tenure length.",
        "business_meaning": "Customers whose smoothed spend differs significantly "
                            "from MonthlyCharges may have upgraded/downgraded services.",
    },
    "ChargesPerTenure": {
        "formula": "MonthlyCharges / max(Tenure, 1)",
        "rationale": "Relative cost pressure per month of tenure.",
        "business_meaning": "High ratio = paying a lot relative to how long they've been a customer.",
    },
    "EstimatedCLV": {
        "formula": "MonthlyCharges * Tenure",
        "rationale": "Approximate total revenue the customer has generated.",
        "business_meaning": "High-CLV customers represent the highest retention priority.",
    },
    "TenureGroup": {
        "formula": "Bins: New (1-12), Growing (13-24), Established (25-48), Loyal (49+)",
        "rationale": "Categorical lifecycle stage from raw continuous tenure.",
        "business_meaning": "New customers (<= 12 months) are at the highest churn risk.",
    },
    "ContractRisk": {
        "formula": "1 if Contract == 'Month-to-month' else 0",
        "rationale": "Month-to-month contracts have zero switching cost.",
        "business_meaning": "Direct churn risk indicator from contract type.",
    },
    "PaymentRisk": {
        "formula": "1 if PaymentMethod == 'Electronic Check' else 0",
        "rationale": "Electronic check users show elevated churn in EDA.",
        "business_meaning": "Payment friction correlates with dissatisfaction.",
    },
    "HighChargeFlag": {
        "formula": "1 if MonthlyCharges > training 75th percentile else 0",
        "rationale": "Flags high-spend customers using a cut learned on training data only.",
        "business_meaning": "High spenders churning = highest revenue impact.",
    },
    "CustomerValueSegment": {
        "formula": "Training terciles of MonthlyCharges: Low / Medium / High",
        "rationale": "Revenue tier from training-only quantile cuts (1/3 and 2/3).",
        "business_meaning": "Different retention budgets warranted by value segment.",
    },
}

CHARGE_FEATURE_COLS = ["HighChargeFlag", "CustomerValueSegment"]


def learn_charge_thresholds(monthly: pd.Series) -> dict:
    """Learn MonthlyCharges cuts from TRAINING rows only."""
    monthly = pd.to_numeric(monthly, errors="coerce").fillna(0)
    return {
        "q33": float(monthly.quantile(1.0 / 3.0)),
        "q66": float(monthly.quantile(2.0 / 3.0)),
        "q75": float(monthly.quantile(0.75)),
    }


def apply_charge_features(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Apply previously learned training cuts. Does not recompute quantiles."""
    if thresholds is None:
        raise ValueError("charge thresholds are required; learn them from training data.")
    df = df.copy()
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    q33 = thresholds["q33"]
    q66 = thresholds["q66"]
    q75 = thresholds["q75"]
    df["HighChargeFlag"] = (monthly > q75).astype(int)
    df["CustomerValueSegment"] = np.where(
        monthly <= q33,
        "Low",
        np.where(monthly <= q66, "Medium", "High"),
    )
    return df


def engineer_features(
    df: pd.DataFrame,
    charge_thresholds=None,
    fit_charge_thresholds: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add engineered features.

    Deterministic features are always computed from the current row.
    HighChargeFlag and CustomerValueSegment are added only when
    `charge_thresholds` is provided, or when `fit_charge_thresholds=True`
    (unit tests). Production must pass training-only thresholds.
    """
    df = df.copy()

    tenure  = pd.to_numeric(df["Tenure"], errors="coerce").fillna(1).clip(lower=1)
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    total   = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    df["AverageMonthlySpend"] = (total / tenure).round(2)
    df["ChargesPerTenure"] = (monthly / tenure).round(4)
    df["EstimatedCLV"] = (monthly * tenure).round(2)
    df["TenureGroup"] = pd.cut(
        tenure,
        bins=[0, 12, 24, 48, 9999],
        labels=["New", "Growing", "Established", "Loyal"],
    ).astype(str)
    df["ContractRisk"] = (
        df["Contract"].astype(str).str.strip() == "Month-to-month"
    ).astype(int)
    df["PaymentRisk"] = (
        df["PaymentMethod"].astype(str).str.strip() == "Electronic Check"
    ).astype(int)

    if charge_thresholds is None and fit_charge_thresholds:
        charge_thresholds = learn_charge_thresholds(df["MonthlyCharges"])
    if charge_thresholds is not None:
        df = apply_charge_features(df, charge_thresholds)

    if verbose:
        print("\n  --- Feature Engineering ---")
        for feat in FEATURE_DOCS:
            if feat in df.columns:
                print(f"  [OK] {feat}")

    return df
