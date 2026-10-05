# Quality Checklist

## Customer Churn Prediction & Retention Intelligence System
## Week 12 Final Capstone

**Verified**: October 2026  
**Pipeline**: COMPLETE (46.1s) | **Tests**: 65/65 PASSED  
**Dataset**: Actual provided customer_churn.csv (500 rows × 9 columns)

---

## Project Overview

| Item | Status | Notes |
|---|---|---|
| Clear business problem statement | PASS | README + business report |
| Business objectives listed | PASS | 5 business + 5 technical objectives |
| Actual provided dataset used | PASS | customer_churn.csv (500 x 9) |
| No fabricated data or results | PASS | All metrics from actual execution |
| Data dictionary exists | PASS | reports/data_dictionary.md |
| Supporting datasets documented | PASS | sales_data.csv + house_prices.csv |

---

## Setup Instructions

| Item | Status | Notes |
|---|---|---|
| requirements.txt exists | PASS | 9 packages, only used ones |
| deployment/requirements.txt exists | PASS | 7 packages for app only |
| Installation in README | PASS | pip install -r requirements.txt |
| `python run_project.py` works | PASS | Completed in 46.1 seconds |
| `python -m pytest -v` works | PASS | 65/65 passed in 9.58s |
| `streamlit run deployment/app.py` works | PASS | Loads real model artefacts |

---

## Code Structure

| Item | Status | Notes |
|---|---|---|
| Modular src/ architecture | PASS | 14 source modules |
| Functions have docstrings | PASS | All public functions documented |
| PEP 8 style | PASS | Clean naming, proper spacing |
| pathlib for all paths | PASS | No hardcoded Windows paths |
| Random state = 42 everywhere | PASS | Set in config.py |
| No data leakage | PASS | Pipeline fitted only on training data |
| CustomerID excluded from features | PASS | Dropped before model input |

---

## Data Pipeline

| Item | Status | Notes |
|---|---|---|
| Data loading from actual files | PASS | src/data_loader.py |
| Data validation (14+ checks) | PASS | 24 PASS, 0 WARN, 0 FAIL |
| Data quality report saved | PASS | outputs/tables/data_quality_report.csv |
| Data cleaning with logging | PASS | src/data_cleaning.py |
| Processed dataset saved | PASS | data/processed/processed_customer_churn.csv |

---

## EDA

| Item | Status | Notes |
|---|---|---|
| EDA figures generated | PASS | 13 figures in outputs/figures/ |
| Churn distribution visualised | PASS | 02_churn_distribution.png |
| Numerical feature distributions | PASS | 03_numerical_distributions.png |
| Categorical feature analysis | PASS | 04–11 figures |
| Correlation analysis | PASS | 12_correlation_matrix.png |
| Key business insights documented | PASS | In notebook + technical doc |

---

## Feature Engineering

| Item | Status | Notes |
|---|---|---|
| >= 5 engineered features | PASS | 8 features created |
| All features documented | PASS | data_dictionary.md + feature_engineering.py |
| Formula documented | PASS | Each feature has formula + rationale |
| No leakage in features | PASS | No target variable used |
| Binary flags for risk factors | PASS | ContractRisk, PaymentRisk, HighChargeFlag |

---

## Preprocessing Pipeline

| Item | Status | Notes |
|---|---|---|
| sklearn ColumnTransformer used | PASS | Numerical + Categorical sub-pipelines |
| Numerical imputation + scaling | PASS | SimpleImputer(median) + StandardScaler |
| Categorical encoding | PASS | OneHotEncoder(handle_unknown='ignore') |
| Stratified train/test split | PASS | 80/20, random_state=42 |
| Pipeline saved to disk | PASS | models/preprocessing_pipeline.pkl |
| Pipeline fitted only on train | PASS | No leakage |

---

## Machine Learning

| Item | Status | Notes |
|---|---|---|
| >= 4 models trained | PASS | Logistic Regression, DT, RF, GB |
| Class imbalance handled | PASS | class_weight='balanced' |
| Cross-validation performed | PASS | 5-fold stratified, ROC-AUC |
| Hyperparameter tuning done | PASS | RandomizedSearchCV, 20 iterations |
| Best params documented | PASS | FINAL_AUDIT.md + technical doc |
| Best model saved | PASS | models/best_model.pkl |

---

## Evaluation

| Item | Status | Notes |
|---|---|---|
| Accuracy reported | PASS | All 4 models |
| Precision reported | PASS | All 4 models |
| Recall reported (primary metric) | PASS | All 4 models |
| F1-Score reported | PASS | All 4 models |
| ROC-AUC reported | PASS | All 4 models |
| PR-AUC reported | PASS | All 4 models |
| Confusion matrices saved | PASS | confusion_matrices.png |
| ROC curves saved | PASS | roc_curves.png |
| PR curves saved | PASS | precision_recall_curves.png |
| Model comparison chart saved | PASS | model_comparison_chart.png |
| Business cost of FP/FN explained | PASS | Technical doc Section 13 |
| Model comparison CSV saved | PASS | outputs/metrics/model_comparison.csv |

---

## Model Interpretation

| Item | Status | Notes |
|---|---|---|
| Feature importances computed | PASS | From model.feature_importances_ |
| Feature importance CSV saved | PASS | outputs/tables/feature_importance.csv |
| Feature importance plot saved | PASS | outputs/figures/feature_importance.png |
| Top churn drivers identified | PASS | Tenure #1 (0.2264) |
| Causation vs correlation noted | PASS | In code + docs + presentation |

---

## Business Output

| Item | Status | Notes |
|---|---|---|
| Customer risk scoring (all 500) | PASS | customer_risk_predictions.csv |
| Risk levels: HIGH/MEDIUM/LOW | PASS | 67 / 9 / 424 customers |
| Risk thresholds documented | PASS | config.py + data dictionary |
| Business recommendations by tier | PASS | business_insights.py |
| Revenue at risk module | PASS | business_analysis.py |
| Retention action plan module | PASS | business_analysis.py |

---

## Deployment

| Item | Status | Notes |
|---|---|---|
| Streamlit app exists | PASS | deployment/app.py |
| App loads real model (no retraining) | PASS | Loads best_model.pkl |
| App validates user inputs | PASS | validate_inputs() tested |
| App shows churn probability | PASS | Metric + progress bar |
| App shows risk level (colour coded) | PASS | Red/Yellow/Green |
| App shows recommendation | PASS | Context-specific text |
| deployment/prediction.py exists | PASS | Single-customer inference |
| src/prediction.py exists | PASS | Batch scoring utility |
| deployment/README.md exists | PASS | Run instructions |
| deployment/requirements.txt exists | PASS | 7 packages |

---

## Testing

| Item | Status | Notes |
|---|---|---|
| test_data.py (10 tests) | PASS | All passed |
| test_deployment.py (10 tests) | PASS | All passed |
| test_features.py (15 tests) | PASS | All passed |
| test_model.py (7 tests) | PASS | All passed |
| test_preprocessing.py (8 tests) | PASS | All passed |
| test_pipeline.py (15 tests) | PASS | All passed |
| **Total: 65 tests** | **ALL PASSED** | **9.58s** |
| Testing evidence report | PASS | reports/testing_evidence.md |

---

## GitHub Readiness

| Item | Status | Notes |
|---|---|---|
| .gitignore exists | PASS | Ignores __pycache__, .venv, .env |
| LICENSE exists | PASS | MIT |
| README.md exists | PASS | Professional GitHub README |
| No secrets in code | PASS | No API keys, passwords |
| No hardcoded paths | PASS | All paths via pathlib from ROOT |
| Clean folder structure | PASS | No unnecessary nested folders |
| Notebook exists | PASS | notebooks/capstone_project.ipynb |

---

## Documentation

| Item | Status | Notes |
|---|---|---|
| Technical documentation (20 sections) | PASS | reports/technical_documentation.md |
| Business report | PASS | reports/business_report.md |
| Executive summary | PASS | reports/executive_summary.md |
| Data dictionary (9+8 cols) | PASS | reports/data_dictionary.md |
| Testing evidence | PASS | reports/testing_evidence.md |
| Interview prep (25 Q&As) | PASS | reports/interview_questions.md |
| Quality checklist | PASS | This file |
| Resume entry | PASS | reports/resume_project_entry.md |
| LinkedIn description | PASS | reports/linkedin_project_description.md |
| Screenshots guide | PASS | reports/screenshots/README.md |
| Presentation (15 slides) | PASS | presentation/presentation.md |
| Speaker notes | PASS | presentation/speaker_notes.md |
| FINAL_AUDIT.md | PASS | All actual metrics |

---

## Summary

| Category | Pass | Fail |
|---|---|---|
| Project Overview | 6 | 0 |
| Setup Instructions | 6 | 0 |
| Code Structure | 6 | 0 |
| Data Pipeline | 5 | 0 |
| EDA | 6 | 0 |
| Feature Engineering | 5 | 0 |
| Preprocessing Pipeline | 6 | 0 |
| Machine Learning | 6 | 0 |
| Evaluation | 11 | 0 |
| Model Interpretation | 5 | 0 |
| Business Output | 6 | 0 |
| Deployment | 10 | 0 |
| Testing | 8 | 0 |
| GitHub Readiness | 6 | 0 |
| Documentation | 13 | 0 |
| **TOTAL** | **105** | **0** |

**RESULT: ALL 105 CHECKS PASS**
