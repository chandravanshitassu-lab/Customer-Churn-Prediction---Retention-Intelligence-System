# Resume Project Entry

## Customer Churn Prediction & Retention Intelligence System

---

### Project Title
**Customer Churn Prediction & Retention Intelligence System**

### Technologies
Python | scikit-learn | Gradient Boosting | pandas | NumPy | matplotlib | seaborn | Streamlit | pytest | joblib | Jupyter Notebook

---

### Resume Bullets

- **Engineered an end-to-end ML pipeline** to predict customer churn using 5 classification algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting), achieving a cross-validation ROC-AUC of **0.7695** on 500 customer records
- **Designed a sklearn ColumnTransformer preprocessing pipeline** with median imputation, StandardScaler, and OneHotEncoder; implemented stratified 80/20 train-test split and 5-fold cross-validation to prevent data leakage and ensure reproducibility
- **Created 9 business-driven engineered features** (Customer Lifetime Value, Contract Risk, Tenure Group, etc.) and performed hyperparameter tuning via RandomizedSearchCV (20 iterations), improving model CV AUC to **0.7552**
- **Deployed a Streamlit web application** that loads saved model artefacts, accepts customer inputs, generates real-time churn probability, and delivers tiered retention recommendations (HIGH/MEDIUM/LOW risk) for 500 customers across 3 risk segments
- **Implemented 45 pytest tests** across data, preprocessing, feature engineering, model, and deployment modules — achieving **100% test pass rate** — and documented the full pipeline in technical, business, and executive report formats

---

### Quantified Outcomes (Actual from Pipeline)

| Metric | Value |
|---|---|
| Cross-validation ROC-AUC | 0.7695 (Logistic Regression) |
| Tuned Gradient Boosting CV AUC | 0.7552 |
| Random Forest Test ROC-AUC | 0.7352 |
| Customers scored | 500 |
| HIGH RISK customers identified | 146 (29.2%) |
| Test pass rate | 45/45 (100%) |
| EDA figures produced | 17 |

> Note: Trained on simulated telecom data for academic purposes.
