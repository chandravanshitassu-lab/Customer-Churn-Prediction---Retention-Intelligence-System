# Screenshots & Visual Documentation

## Customer Churn Prediction & Retention Intelligence System

All generated figures are saved in `outputs/figures/`. Screenshots of the running application and test results should be captured manually.

---

## Generated Figures (Auto-saved by Pipeline)

These files are automatically created when you run `python run_project.py`:

| # | Filename | Description | Location |
|---|---|---|---|
| 1 | `01_dataset_overview.png` | Dataset dimensions, column types, missing value summary | outputs/figures/ |
| 2 | `02_churn_distribution.png` | Pie chart and bar chart of churn class balance (10.6% churn) | outputs/figures/ |
| 3 | `03_numerical_distributions.png` | Histograms for Tenure, MonthlyCharges, TotalCharges (overall + by churn) | outputs/figures/ |
| 4 | `04_categorical_distributions.png` | Count plots for Contract, PaymentMethod, PaperlessBilling | outputs/figures/ |
| 5 | `05_tenure_vs_churn.png` | Tenure distribution by churn + churn rate by tenure group | outputs/figures/ |
| 6 | `06_monthly_charges_vs_churn.png` | Boxplot + histogram of MonthlyCharges by churn status | outputs/figures/ |
| 7 | `07_total_charges_vs_churn.png` | Boxplot + histogram of TotalCharges by churn status | outputs/figures/ |
| 8 | `08_contract_vs_churn.png` | Churn rate (%) and count by contract type | outputs/figures/ |
| 9 | `09_payment_method_vs_churn.png` | Churn rate (%) and count by payment method | outputs/figures/ |
| 10 | `10_paperless_billing_vs_churn.png` | Churn rate (%) and count by paperless billing status | outputs/figures/ |
| 11 | `11_senior_citizen_vs_churn.png` | Churn rate (%) and count by senior citizen status | outputs/figures/ |
| 12 | `12_correlation_matrix.png` | Heatmap of correlation between all numeric features + target | outputs/figures/ |
| 13 | `13_tenure_vs_charges_scatter.png` | Scatter plot: Tenure vs MonthlyCharges, coloured by churn | outputs/figures/ |
| 14 | `confusion_matrices.png` | Confusion matrices for all 4 models side-by-side | outputs/figures/ |
| 15 | `roc_curves.png` | ROC curves for all 4 models with AUC values | outputs/figures/ |
| 16 | `precision_recall_curves.png` | PR curves for all 4 models with PR-AUC values | outputs/figures/ |
| 17 | `model_comparison_chart.png` | Grouped bar chart comparing all models on 5 metrics | outputs/figures/ |
| 18 | `feature_importance.png` | Top 20 feature importances for final model | outputs/figures/ |

---

## Manual Screenshots Required

The following screenshots must be captured manually and saved here:

### 1. Dataset Overview
**How to capture**: Open `outputs/figures/01_dataset_overview.png`
**What to show**: Dataset shape, column types, no missing values

### 2. Churn Distribution
**How to capture**: Open `outputs/figures/02_churn_distribution.png`
**What to show**: 89.4% / 10.6% class imbalance

### 3. Key EDA Insight
**How to capture**: Open `outputs/figures/08_contract_vs_churn.png`
**What to show**: Month-to-month has highest churn rate

### 4. Model Comparison
**How to capture**: Open `outputs/figures/model_comparison_chart.png`
**What to show**: 4 models compared on Accuracy, Precision, Recall, F1, ROC-AUC

### 5. Confusion Matrix
**How to capture**: Open `outputs/figures/confusion_matrices.png`
**What to show**: Random Forest perfect classification on test set

### 6. ROC Curves
**How to capture**: Open `outputs/figures/roc_curves.png`
**What to show**: All models achieve near-perfect AUC

### 7. Feature Importance
**How to capture**: Open `outputs/figures/feature_importance.png`
**What to show**: Tenure dominates with importance 0.2264

### 8. Streamlit Application
**How to capture**:
1. Run: `streamlit run deployment/app.py`
2. Enter sample customer (e.g., Tenure=3, MonthlyCharges=150, Contract=Month-to-month)
3. Take screenshot of prediction result
**What to show**: High Risk prediction with recommendation

### 9. Test Results
**How to capture**:
1. Run: `python -m pytest tests/ -v`
2. Screenshot the terminal output
**What to show**: "65 passed" in green

---

## How to Open Figures

```bash
# View all figures (Windows Explorer)
explorer outputs\figures

# Or open individual files
start outputs\figures\feature_importance.png
start outputs\figures\roc_curves.png
```
