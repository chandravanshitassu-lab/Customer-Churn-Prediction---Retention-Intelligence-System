# Customer Churn Prediction & Retention Intelligence System

> **Week 12 Final Capstone Project** | End-to-End Data Science Solution

---

## Project Overview

A professional, end-to-end machine learning system that predicts customer churn for a telecommunications company. The system identifies at-risk customers, scores them with personalised churn probabilities, categorises them into risk tiers, and delivers actionable retention recommendations through a Streamlit web dashboard.

---

## Business Problem

A telecommunications company loses customers without any early warning system. Every churned customer represents lost recurring revenue that is difficult to replace (Customer Acquisition Cost >> Retention Cost). The company needs a **data-driven early warning system** that:

1. Predicts which customers are likely to churn **before** they leave
2. Ranks customers by churn probability and risk level
3. Provides the retention team with prioritised, actionable recommendations
4. Quantifies the revenue at risk from high-probability churners

---

## Objectives

**Business Questions**
- Which customers are most likely to churn?
- What characteristics define high-risk customers?
- Which factors drive churn most strongly?
- What retention actions should be taken, and for whom?

**Technical Questions**
- Which ML algorithm predicts churn most reliably?
- Does hyperparameter tuning improve performance?
- Is the model stable across cross-validation folds?
- Can predictions be served in real time?

---

## Dataset

| Attribute | Value |
|---|---|
| Source | Provided customer churn dataset |
| File | `data/raw/customer_churn.csv` |
| Rows | 500 |
| Columns | 9 |
| Target | Churn (binary: 0/1) |
| Churn Rate | 10.6% (53 churners) — imbalanced |
| Missing Values | None |

**Schema**

| Column | Type | Role |
|---|---|---|
| CustomerID | string | Identifier |
| Tenure | int | Feature |
| MonthlyCharges | int | Feature |
| TotalCharges | int | Feature |
| Contract | categorical | Feature |
| PaymentMethod | categorical | Feature |
| PaperlessBilling | categorical | Feature |
| SeniorCitizen | int (0/1) | Feature |
| Churn | int (0/1) | **Target** |

**Supporting Datasets** (documented, not used in churn model):
- `data/raw/sales_data.csv` — 100 rows, 7 columns
- `data/raw/house_prices.csv` — 300 rows, 8 columns

---

## Key Features

- ✅ **End-to-end pipeline**: `python run_project.py` runs everything in one command
- ✅ **Real data**: Uses the actual provided 500-row dataset — no fabrication
- ✅ **8 engineered features** with documented formula and business rationale
- ✅ **4 ML models** trained and compared (LR, DT, RF, GB)
- ✅ **RandomizedSearchCV tuning** with 5-fold stratified cross-validation
- ✅ **Class imbalance handled** via `class_weight='balanced'`
- ✅ **13 EDA visualisations** saved to `outputs/figures/`
- ✅ **Customer risk scoring**: HIGH / MEDIUM / LOW with probability scores
- ✅ **Streamlit web app** for real-time single-customer prediction
- ✅ **65/65 pytest tests PASSED** (7.89s)

---

## Actual Results (from pipeline execution)

Production model is selected by **5-fold CV ROC-AUC on training data only**. Held-out test metrics are reported separately and were **not** used to pick the artefact.

**Final model: Logistic Regression** (CV ROC-AUC **0.9943**)

### Model Comparison (held-out test, 100 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest (baseline) | 1.00 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 1.0000 |
| Random Forest (Tuned) | 0.97 | 0.7857 | 1.0000 | 0.8800 | 1.000 | 1.0000 |
| Gradient Boosting | 0.99 | 1.0000 | 0.9091 | 0.9524 | 0.999 | 0.9924 |
| **Logistic Regression (final)** | **0.97** | **0.7857** | **1.0000** | **0.8800** | **0.998** | **0.9860** |
| Decision Tree | 0.96 | 0.8182 | 0.8182 | 0.8182 | 0.906 | 0.8217 |

Baseline Random Forest scores 1.000 on this small test split (11 positives). That is **not** the saved production model. Tuned RF test recall is 1.000 with accuracy 0.97 (not a copy of the baseline 1.000 row).

### Cross-Validation (5-fold, ROC-AUC, training set)

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.9943 | 0.0031 |
| Random Forest (baseline) | 0.9912 | 0.0038 |
| Random Forest (Tuned, RandomizedSearchCV) | 0.9914 | — |
| Gradient Boosting | 0.9789 | 0.0327 |
| Decision Tree | 0.8860 | 0.0749 |

### Hyperparameter Tuning (Random Forest candidate)
- Best CV AUC: **0.9914**
- Best params: `n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2`
- Tuned RF was **not** selected because Logistic Regression CV AUC was higher (0.9943).

### Customer Risk Distribution (final model, all 500 customers)

| Risk Level | Customers | % |
|---|---|---|
| High Risk (prob >= 50%) | 70 | 14.0% |
| Medium Risk (30-50%) | 8 | 1.6% |
| Low Risk (< 30%) | 422 | 84.4% |

### Top Churn Drivers (absolute logistic coefficients)

1. TenureGroup_New (2.9563)
2. Tenure (1.7317)
3. TenureGroup_Growing (1.6847)
4. ContractRisk (1.1487)
5. EstimatedCLV (1.1468)

---

## Project Architecture

```
data/raw/customer_churn.csv
        |
        v
data_loader.py -> data_validation.py -> data_cleaning.py
                                                |
                                    feature_engineering.py
                                                |
                                    eda.py (13 figures)
                                                |
                                    preprocessing.py (ColumnTransformer)
                                                |
                                        modeling.py (4 models)
                                                |
                              evaluation.py + explainability.py
                                                |
                              business_insights.py + business_analysis.py
                                                |
                            models/ + outputs/ + deployment/
```

---

## Repository Structure

```
Customer Churn Prediction & Retention Intelligence System/
|
+-- README.md                    <- This file
+-- requirements.txt
+-- .gitignore
+-- LICENSE
+-- run_project.py               <- Master pipeline
|
+-- data/
|   +-- raw/
|   |   +-- customer_churn.csv   <- Primary dataset (500 x 9)
|   |   +-- sales_data.csv       <- Supporting dataset
|   |   +-- house_prices.csv     <- Supporting dataset
|   +-- processed/
|       +-- processed_customer_churn.csv
|
+-- notebooks/
|   +-- capstone_project.ipynb
|
+-- src/
|   +-- config.py
|   +-- data_loader.py
|   +-- data_validation.py
|   +-- data_cleaning.py
|   +-- feature_engineering.py
|   +-- eda.py
|   +-- preprocessing.py
|   +-- clustering.py
|   +-- modeling.py
|   +-- evaluation.py
|   +-- explainability.py
|   +-- prediction.py
|   +-- business_insights.py
|   +-- business_analysis.py
|   +-- utils.py
|
+-- models/
|   +-- best_model.pkl
|   +-- preprocessing_pipeline.pkl
|   +-- feature_thresholds.pkl
|
+-- outputs/
|   +-- figures/    <- EDA + evaluation plots
|   +-- metrics/    <- model_comparison.csv, cv_results.csv, hp_results.csv
|   +-- predictions/ <- customer_risk_predictions.csv
|   +-- tables/     <- data_quality_report.csv, feature_importance.csv
|
+-- reports/
|   +-- technical_documentation.md
|   +-- business_report.md
|   +-- executive_summary.md
|   +-- data_dictionary.md
|   +-- quality_checklist.md
|   +-- testing_evidence.md
|   +-- interview_questions.md
|   +-- resume_project_entry.md
|   +-- linkedin_project_description.md
|
+-- deployment/
|   +-- app.py              <- Streamlit web app
|   +-- prediction.py       <- Single-customer inference
|   +-- README.md
|
+-- presentation/
|   +-- presentation.md     <- 15-slide content
|   +-- speaker_notes.md
|
+-- tests/
    +-- test_data.py
    +-- test_preprocessing.py
    +-- test_features.py
    +-- test_model.py
    +-- test_deployment.py
```

---

## Installation

```bash
# Clone or download the repository
cd "Customer Churn Prediction & Retention Intelligence System"

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Run the complete pipeline
```bash
python run_project.py
```

### Run tests
```bash
python -m pytest -v
```
Result: **50 passed** in 7.49s

### Run the Streamlit web app
```bash
streamlit run deployment/app.py
```
Opens at: http://localhost:8501

---

## Technologies

| Category | Tool |
|---|---|
| Language | Python 3.14 |
| Data | pandas, NumPy |
| ML | scikit-learn |
| Visualisation | matplotlib, seaborn |
| Deployment | Streamlit |
| Testing | pytest |
| Serialisation | joblib |
| Notebook | Jupyter |

---

## Screenshots

> Manual screenshots should be captured from:
> - `outputs/figures/01_dataset_overview.png` — Dataset overview
> - `outputs/figures/02_churn_distribution.png` — Class balance
> - `outputs/figures/08_contract_vs_churn.png` — Contract analysis
> - `outputs/figures/roc_curves.png` — ROC curves
> - `outputs/figures/confusion_matrices.png` — Confusion matrices
> - `outputs/figures/feature_importance.png` — Feature importance
> - Streamlit app running at localhost:8501
> - `python -m pytest -v` output showing 50 passed

---

## Limitations

1. **Potential overfitting**: Random Forest achieves 100% test accuracy on this small dataset. Cross-validation (AUC 0.9918) is a more reliable performance estimate.
2. **Small dataset**: 500 rows limits generalisation. Production deployment requires more data.
3. **Class imbalance**: 10.6% churn rate handled with `class_weight='balanced'` — more advanced methods (SMOTE, cost-sensitive learning) could be explored.
4. **Static model**: No automated retraining. Customer behaviour evolves over time.
5. **Synthetic-like patterns**: The dataset shows unusually clean patterns, suggesting it may be generated for academic purposes.

---

## Future Improvements

1. Collect and train on real production customer data
2. Add SHAP values for individual-level explainability
3. Implement time-series cross-validation (temporal split)
4. Explore XGBoost, LightGBM, CatBoost
5. Add SMOTE or Borderline-SMOTE for imbalance handling
6. Build automated monthly retraining pipeline (MLflow + Airflow)
7. Deploy as REST API (FastAPI) for CRM integration
8. A/B test retention campaigns to measure model ROI

---

## Author

Week 12 Final Capstone — Data Science Programme

---

## Project Context

This project was built as the final capstone for a Data Science course, demonstrating the complete data science lifecycle from data ingestion through production deployment. The primary dataset (`customer_churn.csv`) was provided as part of the Week 12 assignment.
