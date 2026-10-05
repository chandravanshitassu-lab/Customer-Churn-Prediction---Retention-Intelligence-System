"""
src/business_analysis.py
------------------------
Business analysis module: translates model outputs into actionable
business intelligence for the churn prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_COL, TABLES_DIR


def compute_churn_by_segment(df: pd.DataFrame) -> dict:
    """
    Compute churn rates by each categorical segment.

    Returns
    -------
    dict : {column_name -> pd.Series of churn rates}
    """
    results = {}
    for col in ["Contract", "PaymentMethod", "PaperlessBilling"]:
        if col in df.columns and TARGET_COL in df.columns:
            rate = df.groupby(col)[TARGET_COL].mean() * 100
            results[col] = rate.sort_values(ascending=False)
    return results


def identify_high_risk_profile(df: pd.DataFrame) -> dict:
    """
    Identify the highest-risk customer profile based on EDA findings.

    Returns
    -------
    dict of feature -> highest-risk value
    """
    profile = {}
    if "Contract" in df.columns:
        profile["Contract"] = (
            df.groupby("Contract")[TARGET_COL].mean().idxmax()
        )
    if "PaymentMethod" in df.columns:
        profile["PaymentMethod"] = (
            df.groupby("PaymentMethod")[TARGET_COL].mean().idxmax()
        )
    if "PaperlessBilling" in df.columns:
        profile["PaperlessBilling"] = (
            df.groupby("PaperlessBilling")[TARGET_COL].mean().idxmax()
        )
    if "SeniorCitizen" in df.columns:
        rates = df.groupby("SeniorCitizen")[TARGET_COL].mean()
        profile["SeniorCitizen"] = "Senior" if rates.get(1, 0) > rates.get(0, 0) else "Non-Senior"
    return profile


def estimate_revenue_at_risk(
    predictions_df: pd.DataFrame,
    df_original: pd.DataFrame,
    monthly_charges_col: str = "MonthlyCharges"
) -> pd.DataFrame:
    """
    Estimate monthly revenue at risk for each risk tier.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output from business_insights.generate_predictions()
    df_original : pd.DataFrame
        Original (cleaned) dataset with MonthlyCharges.
    monthly_charges_col : str

    Returns
    -------
    pd.DataFrame
        Revenue at risk by risk level.
    """
    if monthly_charges_col not in df_original.columns:
        return pd.DataFrame()

    merged = predictions_df.copy()
    if monthly_charges_col in df_original.columns:
        merged[monthly_charges_col] = df_original[monthly_charges_col].values

    revenue_summary = (
        merged.groupby("RiskLevel")[monthly_charges_col]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "Customers",
                         "sum":   "Monthly_Revenue_At_Risk",
                         "mean":  "Avg_Monthly_Charges"})
    )
    revenue_summary["Monthly_Revenue_At_Risk"] = (
        revenue_summary["Monthly_Revenue_At_Risk"].round(0)
    )
    revenue_summary["Avg_Monthly_Charges"] = (
        revenue_summary["Avg_Monthly_Charges"].round(2)
    )

    out_path = TABLES_DIR / "revenue_at_risk.csv"
    revenue_summary.to_csv(out_path)
    print(f"  [OK] Revenue at risk saved -> {out_path.name}")
    print(revenue_summary.to_string())
    return revenue_summary


def generate_retention_action_plan(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a customer-level retention action plan CSV.

    Returns
    -------
    pd.DataFrame
    """
    action_map = {
        "High Risk":   "IMMEDIATE: Call within 48h, offer discount/contract upgrade",
        "Medium Risk": "PROACTIVE: Email/SMS engagement campaign, re-score next cycle",
        "Low Risk":    "ROUTINE: Loyalty rewards, upsell premium services",
    }

    plan = predictions_df[["CustomerID", "ChurnProbability", "RiskLevel"]].copy()
    plan["RetentionAction"] = plan["RiskLevel"].map(action_map)

    out_path = TABLES_DIR / "retention_action_plan.csv"
    plan.to_csv(out_path, index=False)
    print(f"  [OK] Retention action plan saved -> {out_path.name}")
    return plan
