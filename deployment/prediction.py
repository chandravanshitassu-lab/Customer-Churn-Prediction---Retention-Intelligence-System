"""
deployment/prediction.py
------------------------
Inference utility: loads saved artefacts and scores a single customer.
Uses the ACTUAL 9-column dataset schema.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import numpy as np

from src.config import (
    BEST_MODEL_PKL, PIPELINE_PKL, FEATURE_THRESHOLDS_PKL,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
)
from src.feature_engineering import engineer_features, apply_charge_features


def load_artefacts():
    """Load and return (preprocessor, model, charge_thresholds)."""
    missing = []
    if not PIPELINE_PKL.exists():
        missing.append(str(PIPELINE_PKL))
    if not BEST_MODEL_PKL.exists():
        missing.append(str(BEST_MODEL_PKL))
    if not FEATURE_THRESHOLDS_PKL.exists():
        missing.append(str(FEATURE_THRESHOLDS_PKL))
    if missing:
        raise FileNotFoundError(
            "Missing artefacts:\n  " + "\n  ".join(missing) +
            "\nRun `python run_project.py` to train and save artefacts."
        )
    return (
        joblib.load(PIPELINE_PKL),
        joblib.load(BEST_MODEL_PKL),
        joblib.load(FEATURE_THRESHOLDS_PKL),
    )


def predict_single_customer(customer_dict: dict) -> dict:
    """
    Generate churn prediction for one customer.

    Parameters
    ----------
    customer_dict : dict
        Keys: Tenure, MonthlyCharges, TotalCharges, Contract,
              PaymentMethod, PaperlessBilling, SeniorCitizen

    Returns
    -------
    dict with keys: prediction, probability, risk_level, recommendation
    """
    preprocessor, model, thresholds = load_artefacts()

    df = apply_charge_features(
        engineer_features(pd.DataFrame([customer_dict]), verbose=False),
        thresholds,
    )

    X_proc   = preprocessor.transform(df)
    prob     = float(model.predict_proba(X_proc)[0, 1])
    pred     = int(model.predict(X_proc)[0])

    if prob >= HIGH_RISK_THRESHOLD:
        risk = "High Risk"
        rec  = (
            "Priority action: Contact this customer within 48 hours. "
            "Offer a personalised discount or upgrade to an annual contract."
        )
    elif prob >= LOW_RISK_THRESHOLD:
        risk = "Medium Risk"
        rec  = (
            "Enrol in proactive engagement campaign. "
            "Send value-reinforcement communications at next billing cycle."
        )
    else:
        risk = "Low Risk"
        rec  = (
            "Customer appears stable. "
            "Consider upselling premium services or enrolling in a loyalty programme."
        )

    return {
        "prediction":    pred,
        "probability":   round(prob, 4),
        "risk_level":    risk,
        "recommendation": rec,
    }


def validate_inputs(inputs: dict) -> list:
    """Return list of validation error messages (empty = valid)."""
    errors = []
    for field, label in [("Tenure", "Tenure"), ("MonthlyCharges", "MonthlyCharges")]:
        val = inputs.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"{label} is required.")
        else:
            try:
                v = float(val)
                if v < 0:
                    errors.append(f"{label} must be >= 0.")
            except (ValueError, TypeError):
                errors.append(f"{label} must be a number.")
    return errors
