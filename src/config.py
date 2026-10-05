"""
config.py
---------
Central configuration for the Customer Churn Prediction & Retention Intelligence System.
All paths, constants, and hyperparameter grids live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root & directory paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

OUTPUTS_DIR     = ROOT_DIR / "outputs"
FIGURES_DIR     = OUTPUTS_DIR / "figures"
METRICS_DIR     = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
TABLES_DIR      = OUTPUTS_DIR / "tables"

MODELS_DIR      = ROOT_DIR / "models"
REPORTS_DIR     = ROOT_DIR / "reports"

# Ensure all output directories exist
for _d in [FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR,
           TABLES_DIR, MODELS_DIR, PROCESSED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset paths  (actual 9-column provided dataset)
# ---------------------------------------------------------------------------
RAW_CHURN_CSV       = RAW_DATA_DIR / "customer_churn.csv"
RAW_SALES_CSV       = RAW_DATA_DIR / "sales_data.csv"
RAW_HOUSE_CSV       = RAW_DATA_DIR / "house_prices.csv"
PROCESSED_CHURN_CSV = PROCESSED_DIR / "processed_customer_churn.csv"

# ---------------------------------------------------------------------------
# Saved artefacts
# ---------------------------------------------------------------------------
BEST_MODEL_PKL          = MODELS_DIR / "best_model.pkl"
PIPELINE_PKL            = MODELS_DIR / "preprocessing_pipeline.pkl"
FEATURE_THRESHOLDS_PKL  = MODELS_DIR / "feature_thresholds.pkl"

# ---------------------------------------------------------------------------
# Dataset schema  (verified against the actual 9-column file)
# ---------------------------------------------------------------------------
TARGET_COL    = "Churn"
ID_COL        = "CustomerID"

NUMERICAL_COLS   = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = ["Contract", "PaymentMethod", "PaperlessBilling"]

EXPECTED_COLUMNS = [ID_COL] + NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL]

VALID_CATEGORIES = {
    "Contract":         {"Month-to-month", "One year", "Two year"},
    "PaymentMethod":    {"Credit Card", "Electronic Check", "Bank Transfer"},
    "PaperlessBilling": {"Yes", "No"},
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------
LOW_RISK_THRESHOLD  = 0.30
HIGH_RISK_THRESHOLD = 0.50   # Tighter threshold due to high class imbalance

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS   = 5
CV_SCORING = "roc_auc"

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}

DT_PARAM_GRID = {
    "max_depth":         [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}

GB_PARAM_GRID = {
    "n_estimators":  [100, 200, 300],
    "learning_rate": [0.05, 0.10, 0.20],
    "max_depth":     [3, 5, 7],
    "subsample":     [0.8, 1.0],
}

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIGURE_DPI = 120
