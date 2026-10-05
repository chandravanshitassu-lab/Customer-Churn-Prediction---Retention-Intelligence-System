"""
modeling.py
-----------
Train multiple classifiers on the churn dataset.
Addresses the severe class imbalance (10.6% positive rate) using
class_weight='balanced' and SMOTE-free resampling.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    RANDOM_STATE, CV_FOLDS, CV_SCORING,
    METRICS_DIR, BEST_MODEL_PKL,
    RF_PARAM_GRID, DT_PARAM_GRID, GB_PARAM_GRID,
)


def build_models() -> dict:
    """Return dict of un-fitted classifiers with class_weight for imbalance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE,
            class_weight="balanced", solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced",
            max_depth=7,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            learning_rate=0.1, max_depth=4,
        ),
    }


def train_all_models(X_train: np.ndarray, y_train: pd.Series) -> dict:
    """Fit every model on the training set."""
    models   = build_models()
    fitted   = {}
    print("\n  --- Model Training ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  [OK] Trained: {name}")
    return fitted


def cross_validate_models(
    models: dict, X_train: np.ndarray, y_train: pd.Series
) -> pd.DataFrame:
    """5-fold stratified cross-validation for each model."""
    print("\n  --- Cross-Validation (5-fold, ROC-AUC) ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring=CV_SCORING, n_jobs=-1)
        rows.append({
            "Model":       name,
            "CV_Mean_AUC": round(scores.mean(), 4),
            "CV_Std_AUC":  round(scores.std(),  4),
            "CV_Min_AUC":  round(scores.min(),  4),
            "CV_Max_AUC":  round(scores.max(),  4),
        })
        print(f"  {name:<30s}  AUC {scores.mean():.4f} +/- {scores.std():.4f}")

    cv_df = pd.DataFrame(rows).sort_values("CV_Mean_AUC", ascending=False)
    out   = METRICS_DIR / "cross_validation_results.csv"
    cv_df.to_csv(out, index=False)
    print(f"  [OK] CV results saved -> {out.name}")
    return cv_df


def tune_model(
    X_train: np.ndarray, y_train: pd.Series,
    model_name: str = "Random Forest",
    n_iter: int = 20,
):
    """
    RandomizedSearchCV tuning for the specified model.

    Returns
    -------
    tuple : (best_model, tuning_df, best_cv_auc)
    """
    print(f"\n  --- Hyperparameter Tuning: {model_name} ---")

    param_grids = {
        "Random Forest":    RF_PARAM_GRID,
        "Decision Tree":    DT_PARAM_GRID,
        "Gradient Boosting": GB_PARAM_GRID,
    }
    base_models = {
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    candidate = base_models.get(model_name, base_models["Random Forest"])
    param_grid = param_grids.get(model_name, RF_PARAM_GRID)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        candidate, param_distributions=param_grid,
        n_iter=n_iter, scoring=CV_SCORING, cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print(f"  Best params  : {search.best_params_}")
    print(f"  Best CV AUC  : {search.best_score_:.4f}")

    results_df = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    out = METRICS_DIR / "hyperparameter_results.csv"
    results_df.to_csv(out, index=False)
    print(f"  [OK] Tuning results saved -> {out.name}")
    return best, results_df, float(search.best_score_)


def save_best_model(model, name: str) -> None:
    """Persist the final model."""
    BEST_MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PKL)
    print(f"\n  [OK] Final model ({name}) saved -> {BEST_MODEL_PKL.name}")
