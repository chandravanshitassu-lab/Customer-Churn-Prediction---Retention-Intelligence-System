# Testing Evidence

## Customer Churn Prediction & Retention Intelligence System

**Date**: October 2026
**Command**: `python -m pytest tests/ -v`
**Result**: **50 passed in 7.49s**

---

## Test Suite Summary

| Module | Tests | Status | Coverage |
|---|---|---|---|
| `tests/test_data.py` | 10 | ALL PASSED | Dataset loading and integrity |
| `tests/test_deployment.py` | 10 | ALL PASSED | Inference and input validation |
| `tests/test_features.py` | 15 | ALL PASSED | Feature engineering correctness |
| `tests/test_model.py` | 7 | ALL PASSED | Training, predictions, artefacts |
| `tests/test_preprocessing.py` | 8 | ALL PASSED | Pipeline, shapes, stratification |
| **TOTAL** | **50** | **50 PASSED** | |

---

## Full Test Output

```
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
rootdir: Customer Churn Prediction & Retention Intelligence System
plugins: anyio-4.14.2
collecting ... collected 50 items

tests/test_data.py::test_dataset_file_exists PASSED               [  2%]
tests/test_data.py::test_dataset_loads PASSED                     [  4%]
tests/test_data.py::test_dataset_not_empty PASSED                 [  6%]
tests/test_data.py::test_expected_columns PASSED                  [  8%]
tests/test_data.py::test_target_column_exists PASSED              [ 10%]
tests/test_data.py::test_target_values_binary PASSED              [ 12%]
tests/test_data.py::test_no_missing_values PASSED                 [ 14%]
tests/test_data.py::test_row_count PASSED                         [ 16%]
tests/test_data.py::test_customer_id_column PASSED                [ 18%]
tests/test_data.py::test_numerical_non_negative PASSED            [ 20%]

tests/test_deployment.py::test_prediction_runs PASSED             [ 22%]
tests/test_deployment.py::test_prediction_has_keys PASSED         [ 24%]
tests/test_deployment.py::test_prediction_binary PASSED           [ 26%]
tests/test_deployment.py::test_probability_range PASSED           [ 28%]
tests/test_deployment.py::test_risk_level_valid PASSED            [ 30%]
tests/test_deployment.py::test_recommendation_not_empty PASSED    [ 32%]
tests/test_deployment.py::test_valid_inputs_pass PASSED           [ 34%]
tests/test_deployment.py::test_missing_tenure_fails PASSED        [ 36%]
tests/test_deployment.py::test_negative_monthly_fails PASSED      [ 38%]
tests/test_deployment.py::test_non_numeric_tenure_fails PASSED    [ 40%]

tests/test_features.py::test_feature_exists[AverageMonthlySpend] PASSED [ 42%]
tests/test_features.py::test_feature_exists[ChargesPerTenure] PASSED    [ 44%]
tests/test_features.py::test_feature_exists[EstimatedCLV] PASSED        [ 46%]
tests/test_features.py::test_feature_exists[TenureGroup] PASSED         [ 48%]
tests/test_features.py::test_feature_exists[ContractRisk] PASSED        [ 50%]
tests/test_features.py::test_feature_exists[PaymentRisk] PASSED         [ 52%]
tests/test_features.py::test_feature_exists[HighChargeFlag] PASSED      [ 54%]
tests/test_features.py::test_feature_exists[CustomerValueSegment] PASSED[ 56%]
tests/test_features.py::test_estimated_clv_non_negative PASSED          [ 58%]
tests/test_features.py::test_average_monthly_spend_non_negative PASSED  [ 60%]
tests/test_features.py::test_contract_risk_binary PASSED                [ 62%]
tests/test_features.py::test_payment_risk_binary PASSED                 [ 64%]
tests/test_features.py::test_high_charge_flag_binary PASSED             [ 66%]
tests/test_features.py::test_tenure_group_valid PASSED                  [ 68%]
tests/test_features.py::test_no_inf_values PASSED                       [ 70%]

tests/test_model.py::test_minimum_four_models PASSED                    [ 72%]
tests/test_model.py::test_predictions_correct_length PASSED             [ 74%]
tests/test_model.py::test_predictions_binary PASSED                     [ 76%]
tests/test_model.py::test_probabilities_in_range PASSED                 [ 78%]
tests/test_model.py::test_saved_model_exists PASSED                     [ 80%]
tests/test_model.py::test_saved_pipeline_exists PASSED                  [ 82%]
tests/test_model.py::test_saved_model_produces_valid_output PASSED      [ 84%]

tests/test_preprocessing.py::test_split_returns_8_tuple PASSED          [ 86%]
tests/test_preprocessing.py::test_train_is_larger_than_test PASSED      [ 88%]
tests/test_preprocessing.py::test_feature_count_consistent PASSED       [ 90%]
tests/test_preprocessing.py::test_no_nan_in_train PASSED                [ 92%]
tests/test_preprocessing.py::test_no_nan_in_test PASSED                 [ 94%]
tests/test_preprocessing.py::test_labels_binary PASSED                  [ 96%]
tests/test_preprocessing.py::test_pipeline_saved PASSED                 [ 98%]
tests/test_preprocessing.py::test_stratification_preserved PASSED       [100%]

============================== 50 passed in 7.49s ==============================
```

---

## What Each Test Verifies

### test_data.py (10 tests)
| Test | Verifies |
|---|---|
| `test_dataset_file_exists` | `data/raw/customer_churn.csv` exists on disk |
| `test_dataset_loads` | `load_churn_data()` returns a DataFrame |
| `test_dataset_not_empty` | DataFrame has > 0 rows and columns |
| `test_expected_columns` | All 9 expected columns are present |
| `test_target_column_exists` | `Churn` column is present |
| `test_target_values_binary` | Churn values are only 0 or 1 |
| `test_no_missing_values` | Zero missing values in dataset |
| `test_row_count` | At least 100 rows |
| `test_customer_id_column` | CustomerID exists and has no duplicates |
| `test_numerical_non_negative` | Tenure, MonthlyCharges, TotalCharges >= 0 |

### test_deployment.py (10 tests)
| Test | Verifies |
|---|---|
| `test_prediction_runs` | `predict_single_customer()` completes without error |
| `test_prediction_has_keys` | Result dict has prediction, probability, risk_level, recommendation |
| `test_prediction_binary` | Prediction is 0 or 1 |
| `test_probability_range` | Probability is in [0.0, 1.0] |
| `test_risk_level_valid` | Risk level is High / Medium / Low Risk |
| `test_recommendation_not_empty` | Recommendation string has > 10 characters |
| `test_valid_inputs_pass` | Valid customer dict produces no validation errors |
| `test_missing_tenure_fails` | None tenure triggers validation error |
| `test_negative_monthly_fails` | Negative MonthlyCharges triggers error |
| `test_non_numeric_tenure_fails` | String tenure triggers error |

### test_features.py (15 tests)
| Test | Verifies |
|---|---|
| `test_feature_exists[X]` × 8 | Each of 8 engineered features exists in DataFrame |
| `test_estimated_clv_non_negative` | EstimatedCLV >= 0 for all rows |
| `test_average_monthly_spend_non_negative` | AverageMonthlySpend >= 0 |
| `test_contract_risk_binary` | ContractRisk is only 0 or 1 |
| `test_payment_risk_binary` | PaymentRisk is only 0 or 1 |
| `test_high_charge_flag_binary` | HighChargeFlag is only 0 or 1 |
| `test_tenure_group_valid` | TenureGroup values are within {New, Growing, Established, Loyal} |
| `test_no_inf_values` | No infinite values in numeric engineered features |

### test_model.py (7 tests)
| Test | Verifies |
|---|---|
| `test_minimum_four_models` | At least 4 models trained |
| `test_predictions_correct_length` | Each model's predictions = len(y_test) |
| `test_predictions_binary` | All models produce only 0/1 predictions |
| `test_probabilities_in_range` | All model probabilities in [0, 1] |
| `test_saved_model_exists` | `models/best_model.pkl` exists |
| `test_saved_pipeline_exists` | `models/preprocessing_pipeline.pkl` exists |
| `test_saved_model_produces_valid_output` | Loaded model produces correct-length probability array |

### test_preprocessing.py (8 tests)
| Test | Verifies |
|---|---|
| `test_split_returns_8_tuple` | `split_and_preprocess()` returns 8-tuple |
| `test_train_is_larger_than_test` | Training set > test set |
| `test_feature_count_consistent` | X_train and X_test have same number of columns |
| `test_no_nan_in_train` | No NaN values in X_train |
| `test_no_nan_in_test` | No NaN values in X_test |
| `test_labels_binary` | y_train and y_test contain only 0 and 1 |
| `test_pipeline_saved` | Pipeline PKL file exists after preprocessing |
| `test_stratification_preserved` | Churn rate in train/test differs by < 10% |
