"""
business_insights.py
--------------------
Customer risk scoring and business recommendations for the 9-column
customer churn dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    TARGET_COL, ID_COL, PREDICTIONS_DIR,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
)


def categorise_risk(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif prob >= LOW_RISK_THRESHOLD:
        return "Medium Risk"
    return "Low Risk"


def generate_predictions(model, preprocessor, df_full, X_raw) -> pd.DataFrame:
    """Score all customers and save predictions CSV."""
    X_proc = preprocessor.transform(X_raw)
    y_prob = model.predict_proba(X_proc)[:, 1]
    y_pred = model.predict(X_proc)

    out = pd.DataFrame({
        "CustomerID":       df_full[ID_COL].values if ID_COL in df_full.columns
                            else range(len(y_prob)),
        "ChurnProbability": np.round(y_prob, 4),
        "Prediction":       y_pred,
        "RiskLevel":        [categorise_risk(p) for p in y_prob],
    })

    if TARGET_COL in df_full.columns:
        out["ActualChurn"] = df_full[TARGET_COL].values

    path = PREDICTIONS_DIR / "customer_risk_predictions.csv"
    out.to_csv(path, index=False)
    print(f"\n  [OK] Predictions saved -> {path.name}")

    print("\n  --- Customer Risk Summary ---")
    for cat in ["High Risk", "Medium Risk", "Low Risk"]:
        n = (out["RiskLevel"] == cat).sum()
        pct = n / len(out) * 100
        print(f"  {cat:<15s}: {n:>4d} customers ({pct:.1f}%)")

    return out


def print_business_recommendations(predictions_df, model_name, comp_df) -> None:
    """Print structured recommendations based on actual risk counts."""
    high   = (predictions_df["RiskLevel"] == "High Risk").sum()
    medium = (predictions_df["RiskLevel"] == "Medium Risk").sum()
    low    = (predictions_df["RiskLevel"] == "Low Risk").sum()
    total  = len(predictions_df)

    recall = roc_auc = "N/A"
    if comp_df is not None and len(comp_df) and "Model" in comp_df.columns:
        match = comp_df[comp_df["Model"] == model_name]
        row = match.iloc[0] if len(match) else comp_df.iloc[0]
        recall  = row.get("Recall", "N/A")
        roc_auc = row.get("ROC_AUC", "N/A")

    print("\n" + "=" * 65)
    print("  BUSINESS RECOMMENDATIONS")
    print("=" * 65)
    print(f"""
  Final Model  : {model_name}
  ROC-AUC      : {roc_auc}
  Recall       : {recall}
  Total Scored : {total} customers

  HIGH RISK ({high} customers, prob >= {HIGH_RISK_THRESHOLD*100:.0f}%)
  -------------------------------------------------------
  * Assign a dedicated retention agent immediately
  * Offer personalised discount or contract upgrade
  * Contact within 48 hours via preferred channel
  * Escalate outstanding service issues to resolution

  MEDIUM RISK ({medium} customers, prob {LOW_RISK_THRESHOLD*100:.0f}%-{HIGH_RISK_THRESHOLD*100:.0f}%)
  -------------------------------------------------------
  * Enrol in proactive email/SMS engagement campaigns
  * Communicate service value and new features
  * Monitor at next billing cycle; re-score monthly

  LOW RISK ({low} customers, prob < {LOW_RISK_THRESHOLD*100:.0f}%)
  -------------------------------------------------------
  * Loyalty programme and reward points
  * Upsell premium tiers or add-on services
  * Referral incentive programme

  KEY METRIC NOTE:
  Recall is the most business-critical metric for churn.
  Missed churners (false negatives) = irrecoverable revenue loss.
  False positives = wasted retention offer (lower cost).
""")
    print("=" * 65)
