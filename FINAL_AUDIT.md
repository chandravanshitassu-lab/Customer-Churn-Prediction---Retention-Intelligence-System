# FINAL AUDIT

## Customer Churn Prediction & Retention Intelligence System
## Week 12 Final Capstone — Upgrade Complete

**Date**: October 2026
**Pipeline**: COMPLETE (46.1 seconds)
**Tests**: 50/50 PASSED (7.49s)
**Dataset**: Actual provided customer_churn.csv (500 rows × 9 columns)

---

## Files Created / Modified

### New Source Modules (added for Week 12)
| File | Status | Description |
|---|---|---|
| `src/config.py` | REWRITTEN | Matches actual 9-column schema |
| `src/data_loader.py` | REWRITTEN | Loads actual CSVs only (no generation) |
| `src/data_validation.py` | REWRITTEN | 14 checks for actual schema |
| `src/data_cleaning.py` | REWRITTEN | Actual dataset cleaning |
| `src/feature_engineering.py` | REWRITTEN | 8 features for 9-column dataset |
| `src/eda.py` | REWRITTEN | 13 figures, actual column names |
| `src/preprocessing.py` | REWRITTEN | ColumnTransformer, stratified split |
| `src/modeling.py` | REWRITTEN | 4 models with class_weight='balanced' |
| `src/evaluation.py` | REWRITTEN | 6 metrics + 4 plots |
| `src/explainability.py` | REWRITTEN | Feature importance + CSV |
| `src/business_insights.py` | REWRITTEN | Risk scoring with correct thresholds |
| `src/utils.py` | NEW | Shared utilities |
| `src/business_analysis.py` | NEW | Revenue at risk, action plan |
| `src/prediction.py` | NEW | Batch prediction utility |
| `src/clustering.py` | NEW | K-Means customer segmentation |
| `run_project.py` | REWRITTEN | 14-step pipeline |

### Test Files
| File | Status | Tests | Result |
|---|---|---|---|
| `tests/test_data.py` | REWRITTEN | 10 | PASSED |
| `tests/test_preprocessing.py` | REWRITTEN | 8 | PASSED |
| `tests/test_features.py` | REWRITTEN | 15 | PASSED |
| `tests/test_model.py` | REWRITTEN | 7 | PASSED |
| `tests/test_deployment.py` | REWRITTEN | 10 | PASSED |

### Deployment
| File | Status |
|---|---|
| `deployment/prediction.py` | REWRITTEN (9-column schema) |
| `deployment/app.py` | REWRITTEN (9-column schema) |
| `deployment/README.md` | EXISTS |

### Datasets in data/raw/
| File | Status | Rows | Cols |
|---|---|---|---|
| `customer_churn.csv` | ACTUAL PROVIDED | 500 | 9 |
| `sales_data.csv` | ACTUAL PROVIDED | 100 | 7 |
| `house_prices.csv` | ACTUAL PROVIDED | 300 | 8 |

### Documentation
| File | Status |
|---|---|
| `README.md` | WRITTEN (actual metrics) |
| `presentation/presentation.md` | WRITTEN (15 slides) |
| `presentation/speaker_notes.md` | WRITTEN (full dialogue) |
| `reports/business_report.md` | WRITTEN |
| `reports/executive_summary.md` | WRITTEN |
| `reports/interview_questions.md` | WRITTEN (25 Q&As) |
| `reports/quality_checklist.md` | WRITTEN |
| `reports/resume_project_entry.md` | WRITTEN |
| `reports/linkedin_project_description.md` | WRITTEN |
| `reports/data_dictionary.md` | EXISTS |
| `reports/technical_documentation.md` | EXISTS |

---

## Actual Model Results (from pipeline run)

### Model Comparison (Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest (Tuned)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Logistic Regression | 0.970 | 0.786 | 1.000 | 0.880 | 0.998 | 0.986 |
| Gradient Boosting | 0.990 | 1.000 | 0.909 | 0.952 | 0.998 | 0.986 |
| Decision Tree | 0.960 | 0.818 | 0.818 | 0.818 | 0.906 | 0.822 |

### Cross-Validation (5-fold, ROC-AUC)

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.9943 | 0.0031 |
| Random Forest | 0.9918 | 0.0033 |
| Gradient Boosting | 0.9789 | 0.0327 |
| Decision Tree | 0.8860 | 0.0749 |

### Hyperparameter Tuning (Random Forest)

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 10 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| **Best CV AUC** | **0.9914** |

### Top Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | Tenure | 0.2264 |
| 2 | TenureGroup_New | 0.1947 |
| 3 | ChargesPerTenure | 0.1646 |
| 4 | EstimatedCLV | 0.1427 |
| 5 | AverageMonthlySpend | 0.1056 |
| 6 | TenureGroup_Loyal | 0.0253 |
| 7 | TenureGroup_Established | 0.0250 |
| 8 | MonthlyCharges | 0.0223 |
| 9 | TenureGroup_Growing | 0.0186 |
| 10 | ContractRisk | 0.0136 |

### Customer Risk Distribution

| Risk Level | Customers | % |
|---|---|---|
| HIGH RISK (prob >= 50%) | 67 | 13.4% |
| MEDIUM RISK (30-50%) | 9 | 1.8% |
| LOW RISK (< 30%) | 424 | 84.8% |

---

## Test Results

```
50 passed in 7.49s
```

| Module | Tests | Status |
|---|---|---|
| test_data.py | 10 | ALL PASS |
| test_deployment.py | 10 | ALL PASS |
| test_features.py | 15 | ALL PASS |
| test_model.py | 7 | ALL PASS |
| test_preprocessing.py | 8 | ALL PASS |
| **TOTAL** | **50** | **50 PASSED** |

---

## Pipeline Steps Completed

| Step | Status | Output |
|---|---|---|
| 1. Load data | PASS | 500 rows x 9 cols |
| 2. Validate data | PASS | 24 PASS, 0 WARN, 0 FAIL |
| 3. Clean data | PASS | processed_customer_churn.csv |
| 4. Feature engineering | PASS | 8 features created |
| 5. EDA | PASS | 13 figures saved |
| 6. Preprocessing | PASS | Pipeline saved |
| 7. Train models | PASS | 4 models trained |
| 8. Evaluate models | PASS | model_comparison.csv + 4 plots |
| 9. Cross-validate | PASS | cross_validation_results.csv |
| 10. Hyperparameter tuning | PASS | Best CV AUC: 0.9914 |
| 11. Select & save model | PASS | best_model.pkl |
| 12. Feature importance | PASS | feature_importance.csv + .png |
| 13. Risk scoring | PASS | customer_risk_predictions.csv |
| 14. Business recommendations | PASS | Printed to console |

---

## Deployment Status

| Component | Status |
|---|---|
| `streamlit run deployment/app.py` | READY — loads PKL artefacts |
| `deployment/prediction.py` | READY — tested, 10/10 tests pass |
| Model artefacts | SAVED — `models/best_model.pkl` |
| Pipeline artefacts | SAVED — `models/preprocessing_pipeline.pkl` |

---

## Week 12 Requirements Checklist

| Requirement | Status |
|---|---|
| End-to-end workflow | PASS |
| Professional documentation | PASS |
| Organized GitHub structure | PASS |
| Business presentation (15 slides) | PASS |
| Business recommendations | PASS |
| Basic model deployment (Streamlit) | PASS |
| Technical documentation | PASS |
| Business report | PASS |
| Web interface / dashboard | PASS |
| Testing evidence (50 tests) | PASS |
| Visual documentation (27 figures) | PASS |
| Career / interview preparation | PASS |
| Actual provided datasets used | PASS |
| No fabricated results | PASS |

---

## Notes

- **100% test accuracy**: The Random Forest achieves perfect test scores on this small 500-row dataset. The cross-validation AUC of 0.9918 is the more reliable generalisation estimate. This limitation is documented in the README, technical documentation, and presentation.
- **Dataset**: The actual provided `customer_churn data.csv` (with space in filename) was copied to `data/raw/customer_churn.csv`. No data was modified.
- **Supporting datasets**: `sales_data.csv` and `house_prices.csv` are documented in the data dictionary but NOT used in the churn model (correct per instructions).

---

## FINAL STATUS: READY FOR SUBMISSION
