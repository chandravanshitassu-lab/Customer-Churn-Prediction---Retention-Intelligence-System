# Technical Documentation

## Customer Churn Prediction & Retention Intelligence System
### Week 12 Final Capstone

**Version**: 2.0 (Week 12 Upgrade)
**Date**: October 2026
**Python**: 3.14 | **Framework**: scikit-learn

---

## 1. Project Overview

This project implements a complete, end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. The system ingests raw customer data, validates and cleans it, engineers business-relevant features, trains and tunes multiple classification models, evaluates them with comprehensive metrics, and deploys the final model as a Streamlit web application.

The project follows the complete Data Science lifecycle:
**Data → Validation → Cleaning → EDA → Feature Engineering → Preprocessing → Modelling → Tuning → Evaluation → Interpretation → Deployment → Testing → Documentation**

---

## 2. Business Problem

**Context**: A telecommunications company with 500 customer records experiences a 10.6% churn rate. The retention team currently has no systematic early warning system — churn is detected after it occurs.

**Problem Statement**: Predict which customers are most likely to churn in the next billing cycle, enabling proactive retention intervention before revenue is lost.

**Business Cost of Churn**:
- Each churned customer = lost recurring monthly revenue (\$20–\$199/month)
- Customer Acquisition Cost >> Customer Retention Cost
- 53 customers churning = up to ~\$10,547/month maximum revenue exposure

---

## 3. Objectives

| Objective | Type | Status |
|---|---|---|
| Predict churn with ROC-AUC > 0.90 | Technical | ACHIEVED (0.9914 CV) |
| Achieve Recall > 0.85 | Technical | ACHIEVED (1.00 RF) |
| Score all customers with risk level | Business | ACHIEVED |
| Deploy real-time prediction app | Technical | ACHIEVED |
| Pass automated test suite | QA | ACHIEVED (50/50) |

---

## 4. Dataset

**File**: `data/raw/customer_churn.csv`
**Source**: Provided dataset (actual customer records)
**Rows**: 500 | **Columns**: 9

**Class Distribution**:
- Not Churned (0): 447 (89.4%)
- Churned (1): 53 (10.6%)
- **Imbalance ratio**: ~8.4:1

**Supporting Datasets** (not used in model):
- `data/raw/sales_data.csv` — 100 rows, 7 cols
- `data/raw/house_prices.csv` — 300 rows, 8 cols

---

## 5. Data Dictionary

See `reports/data_dictionary.md` for full documentation of all 9 original features and 8 engineered features.

Key column roles:
- **Identifier**: CustomerID (excluded from model)
- **Target**: Churn (0/1)
- **Numerical features**: Tenure, MonthlyCharges, TotalCharges, SeniorCitizen
- **Categorical features**: Contract, PaymentMethod, PaperlessBilling

---

## 6. Data Validation

**Module**: `src/data_validation.py`
**Output**: `outputs/tables/data_quality_report.csv`

**Checks performed** (24 total):
1. File exists on disk
2. Row count meets minimum threshold
3. Column count validation
4. All expected columns present
5. Target column exists
6. Target values are strictly binary (0/1)
7. Churn rate is within plausible range
8. Missing values count and location
9. Duplicate row detection
10. Duplicate CustomerID detection
11. Non-negative values for Tenure
12. Non-negative values for MonthlyCharges
13. Non-negative values for TotalCharges
14. SeniorCitizen is binary (0/1)
15–17. Valid categories for Contract, PaymentMethod, PaperlessBilling
18–21. Correct numeric dtypes for all numerical columns
22–24. Extreme outlier detection (3×IQR) for Tenure, MonthlyCharges, TotalCharges

**Result**: 24 PASS | 0 WARN | 0 FAIL

---

## 7. Data Cleaning

**Module**: `src/data_cleaning.py`
**Output**: `data/processed/processed_customer_churn.csv`

**Transformations applied**:
1. Strip whitespace from column names
2. Strip whitespace from all string values
3. Standardise categorical value casing (pre-validated as consistent)
4. Coerce numerical columns to float (with NaN detection)
5. Median imputation for any missing numerical values (0 found in actual data)
6. Mode imputation for any missing categorical values (0 found)
7. Remove exact duplicate rows (0 found)
8. Clip numerical columns to valid ranges (no clipping needed)
9. Ensure Churn and SeniorCitizen are integer type

**Why clean if data is already clean?** The cleaning pipeline is designed to be robust to real-world data issues. Even with a clean academic dataset, the pipeline ensures reproducibility and handles edge cases that would appear in production.

---

## 8. Exploratory Data Analysis

**Module**: `src/eda.py`
**Output**: 13 figures in `outputs/figures/`

| Figure | File | Key Insight |
|---|---|---|
| Dataset Overview | 01_dataset_overview.png | Shape, types, missing values |
| Churn Distribution | 02_churn_distribution.png | 10.6% churn rate (imbalanced) |
| Numerical Distributions | 03_numerical_distributions.png | Tenure bimodal, charges right-skewed |
| Categorical Distributions | 04_categorical_distributions.png | Contract type balance |
| Tenure vs Churn | 05_tenure_vs_churn.png | New customers churn most |
| MonthlyCharges vs Churn | 06_monthly_charges_vs_churn.png | Higher charges = more churn |
| TotalCharges vs Churn | 07_total_charges_vs_churn.png | Wide spread for churners |
| Contract vs Churn | 08_contract_vs_churn.png | Month-to-month highest churn |
| PaymentMethod vs Churn | 09_payment_method_vs_churn.png | Electronic Check elevated |
| PaperlessBilling vs Churn | 10_paperless_billing_vs_churn.png | Yes slightly higher churn |
| SeniorCitizen vs Churn | 11_senior_citizen_vs_churn.png | Seniors slightly higher risk |
| Correlation Matrix | 12_correlation_matrix.png | Tenure negatively correlated with Churn |
| Scatter: Tenure vs Charges | 13_tenure_vs_charges_scatter.png | Churn concentrated in short-tenure, high-charge segment |

---

## 9. Feature Engineering

**Module**: `src/feature_engineering.py`

8 business-driven features created. Documented in `reports/data_dictionary.md`.

**Key principles**:
- No features use the target variable (no leakage)
- No features use future data
- All features have documented business rationale
- Binary flags derive from known high-risk categories

---

## 10. Preprocessing Pipeline

**Module**: `src/preprocessing.py`
**Output**: `models/preprocessing_pipeline.pkl`

**Architecture** (sklearn `ColumnTransformer`):

```
ColumnTransformer
├── Numerical pipeline
│   ├── SimpleImputer(strategy='median')
│   └── StandardScaler()
└── Categorical pipeline
    ├── SimpleImputer(strategy='most_frequent')
    └── OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

**Numerical columns** (7): Tenure, MonthlyCharges, TotalCharges, SeniorCitizen, AverageMonthlySpend, ChargesPerTenure, EstimatedCLV + binary flags

**Categorical columns** (5): Contract, PaymentMethod, PaperlessBilling, TenureGroup, CustomerValueSegment

**Train/test split**:
- 80% training: 400 rows (42 churners)
- 20% test: 100 rows (11 churners)
- Stratified split preserves churn ratio
- `random_state=42` for reproducibility

**Leakage prevention**: Pipeline is fitted ONLY on training data. Test data is transformed using the training-fitted parameters.

---

## 11. Machine Learning Algorithms

| Algorithm | Module | Class Weight | Rationale |
|---|---|---|---|
| Logistic Regression | sklearn | balanced | Linear baseline; interpretable coefficients; fast |
| Decision Tree | sklearn | balanced | Non-linear baseline; human-readable rules |
| Random Forest | sklearn | balanced | Bagging ensemble; robust to noise; strong default |
| Gradient Boosting | sklearn | implicit | Boosting ensemble; sequential error correction |

**Why `class_weight='balanced'`?**
With 10.6% positive class, a naive classifier predicts "no churn" for all inputs and achieves 89.4% accuracy while having 0% recall on the target class. `class_weight='balanced'` automatically adjusts sample weights so the minority class (churners) receives proportionally higher weight during training.

---

## 12. Hyperparameter Tuning

**Algorithm**: `RandomizedSearchCV`
**Model tuned**: Random Forest
**Folds**: 5-fold Stratified K-Fold
**Iterations**: 20
**Scoring**: ROC-AUC

**Search space**:
```python
{
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
```

**Best parameters found**:
```python
{
    "n_estimators":      200,
    "max_depth":         10,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
}
```

**Best CV AUC**: 0.9914

**Why RandomizedSearchCV over GridSearchCV?** The search space has 3 × 4 × 3 × 3 = 108 combinations. RandomizedSearchCV samples 20 random combinations instead of exhaustively trying all 108 × 5 folds = 540 fits, making it more efficient without significant loss in quality.

---

## 13. Evaluation Metrics

**Module**: `src/evaluation.py`
**Outputs**: `outputs/metrics/`, `outputs/figures/`

| Metric | Formula | Business Meaning |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness — misleading with imbalance |
| Precision | TP/(TP+FP) | Of predicted churners, how many actually churned |
| Recall | TP/(TP+FN) | Of actual churners, how many were caught |
| F1-Score | 2×(P×R)/(P+R) | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under ROC curve | Model's ranking ability across thresholds |
| PR-AUC | Area under PR curve | Performance on positive class (preferred for imbalance) |

**Business Cost Matrix**:
| Prediction | Actual | Type | Business Cost |
|---|---|---|---|
| No Churn | Churn | False Negative | HIGH — customer leaves permanently |
| Churn | No Churn | False Positive | LOW — wasted retention offer |
| No Churn | No Churn | True Negative | Zero cost |
| Churn | Churn | True Positive | Negative cost (retention saves revenue) |

**Conclusion**: Recall must be maximised over Accuracy because the cost of missing a churner far exceeds the cost of a false alarm.

---

## 14. Model Selection

**Final Model**: Random Forest (Tuned)

**Selection criteria**:
1. Highest ROC-AUC on test set (1.000)
2. Highest CV AUC (0.9914 after tuning)
3. Lowest CV standard deviation (0.0033) — most stable
4. Perfect Recall (1.000) — zero missed churners on test

**Note on 100% test accuracy**: The Random Forest achieves perfect scores on the 100-row test set. With only 11 churners in the test split, the model may be benefitting from the strong patterns in this dataset. The 5-fold CV AUC of 0.9914 is a more robust estimate of generalisation performance.

---

## 15. Model Interpretation

**Module**: `src/explainability.py`
**Outputs**: `outputs/tables/feature_importance.csv`, `outputs/figures/feature_importance.png`

Feature importance extracted from `model.feature_importances_` (Random Forest native).

**Top 5 drivers**:
1. Tenure (0.2264) — How long the customer has been with the company
2. TenureGroup_New (0.1947) — Being a new customer (≤12 months)
3. ChargesPerTenure (0.1646) — Cost pressure relative to relationship length
4. EstimatedCLV (0.1427) — Total estimated revenue from the customer
5. AverageMonthlySpend (0.1056) — Smoothed monthly spend

**Important caveat**: Feature importance shows statistical association with churn, NOT causal direction. High charges are associated with churn — this does not mean that reducing charges will necessarily reduce churn.

---

## 16. Customer Risk Scoring

**Module**: `src/business_insights.py`
**Output**: `outputs/predictions/customer_risk_predictions.csv`

**Risk thresholds** (documented in `src/config.py`):
- HIGH RISK: Churn probability ≥ 50%
- MEDIUM RISK: 30% ≤ probability < 50%
- LOW RISK: Probability < 30%

**Results on 500 customers**:
- HIGH RISK: 67 (13.4%)
- MEDIUM RISK: 9 (1.8%)
- LOW RISK: 424 (84.8%)

**Output columns**: CustomerID, ChurnProbability, Prediction, RiskLevel, ActualChurn

---

## 17. Deployment Architecture

```
User Input
    |
Streamlit Interface (deployment/app.py)
    |
Input Validation (validate_inputs())
    |
Feature Engineering (engineered in-memory)
    |
preprocessing_pipeline.pkl (ColumnTransformer)
    |
best_model.pkl (Random Forest)
    |
Churn Probability (predict_proba)
    |
Risk Classification (HIGH / MEDIUM / LOW)
    |
Retention Recommendation
    |
Dashboard Output
```

**Key design decision**: The model is loaded once at startup (or on first prediction), not retrained. This ensures fast inference and reproducible results.

**Run command**: `streamlit run deployment/app.py`

---

## 18. Testing

**Framework**: pytest 9.1.1
**Total tests**: 65 (50 unit + 15 pipeline integration)
**Result**: All passed

See `reports/testing_evidence.md` for full test output and test-by-test descriptions.

---

## 19. Limitations

1. **Dataset size**: 500 rows is small for a production ML model. Cross-validation (CV AUC 0.9914) is more reliable than the 100-row test result.
2. **Static model**: No automated retraining. Customer behaviour changes over time — model performance will degrade without periodic retraining.
3. **Class imbalance handling**: `class_weight='balanced'` is a simple approach. More advanced methods (SMOTE, cost-sensitive learning, threshold optimisation) could improve the precision-recall tradeoff.
4. **No causal inference**: Feature importance shows correlation, not causation. Retention decisions should not be based solely on model outputs.
5. **Feature drift**: Engineered features like ChargesPerTenure assume stable billing patterns. Major pricing changes would require feature recalculation.

---

## 20. Future Improvements

| Improvement | Priority | Complexity |
|---|---|---|
| Collect real production data (1000+ rows) | HIGH | LOW |
| Add SHAP values for individual-level explanation | HIGH | MEDIUM |
| Implement temporal cross-validation | HIGH | MEDIUM |
| Deploy REST API (FastAPI) for CRM integration | MEDIUM | MEDIUM |
| Add automated monthly retraining pipeline | MEDIUM | HIGH |
| Explore XGBoost, LightGBM, CatBoost | LOW | LOW |
| A/B test retention campaigns | HIGH | HIGH |
| Add calibration (Platt scaling) for probability reliability | MEDIUM | LOW |
