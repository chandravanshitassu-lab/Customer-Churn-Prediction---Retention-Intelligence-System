"""
src/prediction.py
-----------------
Batch prediction utility: scores a dataset of new customers using
the saved model and preprocessing pipeline.

This is separate from deployment/prediction.py (which handles single-customer
inference for the Streamlit app).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    BEST_MODEL_PKL, PIPELINE_PKL, FEATURE_THRESHOLDS_PKL,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
    TARGET_COL, ID_COL, PREDICTIONS_DIR,
)
from src.feature_engineering import engineer_features, apply_charge_features


def score_new_customers(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of new customers.

    Parameters
    ----------
    df_new : pd.DataFrame
        New customers with same schema as training data
        (excluding Churn column).

    Returns
    -------
    pd.DataFrame
        CustomerID, ChurnProbability, Prediction, RiskLevel, RetentionAction
    """
    if not PIPELINE_PKL.exists() or not BEST_MODEL_PKL.exists():
        raise FileNotFoundError(
            "Saved artefacts not found. Run `python run_project.py` first."
        )
    if not FEATURE_THRESHOLDS_PKL.exists():
        raise FileNotFoundError(
            "Feature thresholds not found. Run `python run_project.py` first."
        )

    preprocessor = joblib.load(PIPELINE_PKL)
    model        = joblib.load(BEST_MODEL_PKL)
    thresholds   = joblib.load(FEATURE_THRESHOLDS_PKL)

    df_feat = apply_charge_features(
        engineer_features(df_new, verbose=False), thresholds
    )

    drop_cols = [c for c in [ID_COL, TARGET_COL] if c in df_feat.columns]
    X         = df_feat.drop(columns=drop_cols)

    X_proc = preprocessor.transform(X)
    probs  = model.predict_proba(X_proc)[:, 1]
    preds  = model.predict(X_proc)

    def _risk(p):
        if p >= HIGH_RISK_THRESHOLD:
            return "High Risk"
        elif p >= LOW_RISK_THRESHOLD:
            return "Medium Risk"
        return "Low Risk"

    action_map = {
        "High Risk":   "Call within 48h, offer contract upgrade",
        "Medium Risk": "Email engagement campaign, re-score monthly",
        "Low Risk":    "Loyalty programme, upsell opportunity",
    }

    results = pd.DataFrame({
        "CustomerID":       df_new[ID_COL].values if ID_COL in df_new.columns
                            else range(len(preds)),
        "ChurnProbability": np.round(probs, 4),
        "Prediction":       preds,
        "RiskLevel":        [_risk(p) for p in probs],
    })
    results["RetentionAction"] = results["RiskLevel"].map(action_map)

    out = PREDICTIONS_DIR / "new_customer_predictions.csv"
    results.to_csv(out, index=False)
    print(f"  [OK] Scored {len(results)} customers -> {out.name}")
    return results


if __name__ == "__main__":
    # Example: score the full training dataset as a demo
    from src.data_loader   import load_churn_data
    from src.data_cleaning import clean_data
    df = load_churn_data()
    df = clean_data(df, verbose=False)
    scored = score_new_customers(df)
    print(scored.head())
