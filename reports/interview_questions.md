# Interview Preparation Guide

## Customer Churn Prediction & Retention Intelligence System

---

### Q1. Can you walk me through this project?

**Answer**: This is an end-to-end machine learning system to predict customer churn for a telecommunications company. I built the complete data science pipeline — from synthetic data generation and validation through EDA, feature engineering, model training, evaluation, hyperparameter tuning, explainability, and deployment as a Streamlit web application. The final model is a tuned Gradient Boosting classifier that predicts churn probability and categorises customers into HIGH, MEDIUM, and LOW risk tiers with tailored retention recommendations.

---

### Q2. What was the business problem?

**Answer**: The company had no early warning system for churn. Customers were leaving without any advance signal, meaning retention efforts were purely reactive. The goal was to predict churn _before_ it happens so the retention team can intervene within a critical 30-day window.

---

### Q3. What dataset did you use?

**Answer**: I used a synthetic 500-row telecom customer dataset that mirrors realistic churn patterns from public benchmarks like the IBM Telco dataset. It has 16 features covering demographics, services, contract type, billing, and payment method, with a binary churn target. I clearly documented it as simulated data throughout the project.

---

### Q4. How did you handle missing values?

**Answer**: I used a two-strategy approach in an sklearn Pipeline: median imputation for numerical columns (Tenure, MonthlyCharges, TotalCharges) and mode imputation for categorical columns. Critically, I fitted the imputer **only on training data** and applied it to test data to prevent data leakage. About 3% of numerical values were missing (introduced artificially to make the exercise realistic).

---

### Q5. What feature engineering did you do?

**Answer**: I created 9 business-oriented engineered features:
1. Customer Lifetime Value (MonthlyCharges × Tenure)
2. Average Monthly Value (TotalCharges / Tenure)
3. Tenure Group (New/Growing/Loyal/Champion bins)
4. Charge per Tenure (relative cost pressure)
5. High Value Customer (above 75th percentile monthly charges)
6. Short Tenure Flag (≤ 6 months)
7. Payment Risk (Electronic check indicator)
8. Contract Risk (Month-to-month indicator)
9. Customer Value Segment (Low/Medium/High tercile)

Each was documented with its formula, rationale, and business meaning.

---

### Q6. How did you build the preprocessing pipeline?

**Answer**: I used sklearn's `ColumnTransformer` with two sub-pipelines:
- **Numerical**: `SimpleImputer(median)` → `StandardScaler`
- **Categorical**: `SimpleImputer(mode)` → `OneHotEncoder(handle_unknown='ignore')`

The pipeline was fitted only on training data to prevent leakage. I saved the fitted pipeline with joblib and reproduced it identically during inference in the deployment module.

---

### Q7. Which ML algorithms did you use and why?

**Answer**: I trained 5 models:
- **Logistic Regression** — Linear baseline, interpretable coefficients
- **Decision Tree** — Non-linear baseline, human-readable rules
- **Random Forest** — Ensemble bagging, handles high-dimensional data well
- **Gradient Boosting** — Sequential error correction, typically strongest performer
- **HistGradientBoosting** — Faster variant with native missing value support

I used ensemble methods because they handle feature interactions and non-linearities common in customer behaviour data.

---

### Q8. How did you select the final model?

**Answer**: I compared all 5 models on ROC-AUC, Precision, Recall, F1, and PR-AUC on a held-out test set, then validated stability with 5-fold stratified cross-validation. I selected Gradient Boosting as the candidate for tuning because it had the highest ROC-AUC. After hyperparameter tuning with RandomizedSearchCV, the tuned model became the final model. I did NOT select purely on accuracy — recall was weighted heavily because missing a churner (false negative) is more costly than a false alarm.

---

### Q9. Why is Recall more important than Precision for churn prediction?

**Answer**: In churn prediction:
- **False negative** = predicting "no churn" when the customer will churn → we lose the customer permanently
- **False positive** = predicting "churn" for someone who won't → we waste a retention offer, but the customer stays

The cost of missing a churner far exceeds the cost of an unnecessary retention call. Therefore, I optimised for high recall while monitoring precision to avoid wasting too many resources on low-risk customers.

---

### Q10. What is ROC-AUC and what does it mean for this project?

**Answer**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve) measures the model's ability to discriminate between churners and non-churners across all possible classification thresholds. A score of 1.0 = perfect; 0.5 = random. For churn prediction, ROC-AUC is the primary metric because it evaluates the model's ranking ability — how well it ranks actual churners above non-churners — which is what matters for prioritising retention outreach.

---

### Q11. What is PR-AUC and when is it preferred?

**Answer**: PR-AUC (Precision-Recall Area Under Curve) focuses specifically on the positive class (churners). It's preferred over ROC-AUC when the dataset is imbalanced because ROC-AUC can be overly optimistic on imbalanced data. For this dataset with ~30-40% churn rate, ROC-AUC is reasonable, but I reported both to give a complete picture.

---

### Q12. How did you handle class imbalance?

**Answer**: The dataset had ~30-40% churners (moderate imbalance). I used:
1. `class_weight="balanced"` in Logistic Regression, Decision Tree, and Random Forest
2. Stratified train-test split to preserve the churn ratio
3. 5-fold stratified cross-validation
4. Evaluated with PR-AUC in addition to ROC-AUC

I did not use SMOTE to keep dependencies minimal, but it would be a valid enhancement.

---

### Q13. How did you do cross-validation?

**Answer**: I used `StratifiedKFold(n_splits=5)` with ROC-AUC as the scoring metric. Stratified CV was important to ensure each fold maintained the same churn ratio as the full dataset. I reported mean AUC and standard deviation across folds. A low standard deviation indicates stable model performance, reducing the risk of overfitting.

---

### Q14. How did you do hyperparameter tuning?

**Answer**: I used `RandomizedSearchCV` with 20 iterations and 5-fold stratified CV. The parameters tuned for Gradient Boosting were:
- `n_estimators`: [100, 200]
- `learning_rate`: [0.05, 0.10, 0.20]
- `max_depth`: [3, 5, 7]
- `subsample`: [0.8, 1.0]

I chose `RandomizedSearchCV` over `GridSearchCV` because it samples the parameter space more efficiently when the grid is large. Results were saved to `outputs/metrics/hyperparameter_results.csv`.

---

### Q15. How did you interpret the model?

**Answer**: I extracted feature importances from `model.feature_importances_` (available for tree-based models). I plotted the top 20 features ranked by importance and explained them in business language. I clearly documented that importance indicates correlation with churn, not causation — a common interview follow-up.

---

### Q16. How did you deploy the model?

**Answer**: I built a Streamlit web application (`deployment/app.py`). It loads the saved preprocessing pipeline and model, accepts customer inputs through a form, applies the same feature engineering as training, preprocesses the inputs, generates a churn probability, and displays the risk category with a personalised recommendation. The prediction is always generated by the actual model — never hard-coded.

---

### Q17. How did you test the project?

**Answer**: I wrote 25+ pytest tests across 5 modules:
- `test_data.py` — Dataset existence, shape, columns, target validity
- `test_preprocessing.py` — Pipeline output shape, no NaN leakage, binary labels
- `test_features.py` — Engineered feature existence, non-negativity, binary flags, valid categories
- `test_model.py` — Training, prediction shape/binary values, probability range [0,1]
- `test_deployment.py` — Prediction function, input validation for edge cases

---

### Q18. What is data leakage and how did you prevent it?

**Answer**: Data leakage occurs when information from the test set (or future data) is used during training, causing unrealistically optimistic evaluation. I prevented it by:
1. Fitting the preprocessing pipeline ONLY on training data
2. Applying the fitted pipeline (without refitting) to test data
3. Computing feature importance only after final model training
4. Not using the test set for any model selection decisions (only for final evaluation)

---

### Q19. What would you do differently with more time?

**Answer**: 
1. Use real production data instead of synthetic data
2. Add SHAP values for individual-level explainability
3. Implement time-series cross-validation (customers change over time)
4. Explore XGBoost, LightGBM for potentially better performance
5. Add SMOTE or class-weighted cost functions
6. Build an automated monthly retraining pipeline
7. Measure retention campaign ROI with A/B testing

---

### Q20. How would you explain this project to a non-technical stakeholder?

**Answer**: "We built a system that learns from historical customer data to identify which customers are most likely to cancel their service in the next month. Think of it as a weather forecast, but for customer loyalty. It gives each customer a 'churn score' from 0-100%. Customers above 60% are flagged as HIGH RISK and should be contacted immediately with a retention offer. The system updates automatically and can be used by anyone through a simple web interface."

---

### Q21. What is the difference between Gradient Boosting and Random Forest?

**Answer**: Both are ensemble methods, but they differ in how they combine trees:
- **Random Forest**: Trains many trees in parallel on random subsets (bagging), then averages predictions. Robust to overfitting.
- **Gradient Boosting**: Trains trees sequentially, where each tree corrects the errors of the previous one. Often achieves higher accuracy but requires more careful tuning and is more prone to overfitting if not regularised.

For churn prediction, Gradient Boosting typically performs better because it iteratively focuses on the hardest-to-classify customers.

---

### Q22. What does StandardScaler do and why is it needed?

**Answer**: StandardScaler transforms numerical features to have mean=0 and standard deviation=1. It's needed because:
1. Logistic Regression and distance-based algorithms are sensitive to feature scale
2. Without scaling, features with large values (e.g., TotalCharges in hundreds) would dominate features with small values (e.g., SeniorCitizen 0/1)
3. Scaling is fitted ONLY on training data — using training statistics to transform test data prevents leakage

Note: Tree-based models (Random Forest, Gradient Boosting) don't technically require scaling, but including it in the pipeline ensures consistent preprocessing regardless of model choice.

---

### Q23. How would you evaluate if the model is production-ready?

**Answer**: I'd check:
1. **Performance on new data**: Evaluate on a completely held-out time window
2. **Business metric alignment**: Measure actual retention improvement (A/B test)
3. **Calibration**: Ensure predicted probabilities match observed churn rates (reliability diagram)
4. **Fairness**: Check for demographic bias in predictions
5. **Latency**: Ensure predictions are fast enough for real-time use
6. **Concept drift**: Monitor if model performance degrades over time as customer behaviour changes
7. **Data quality**: Ensure production data has the same schema as training data

---

### Q24. What is the F1-score and when would you use it as your primary metric?

**Answer**: F1-score is the harmonic mean of Precision and Recall: F1 = 2 × (Precision × Recall) / (Precision + Recall). It balances both false positives and false negatives. I would use it as primary metric when:
- The cost of false positives and false negatives is approximately equal
- The dataset is moderately imbalanced
For churn, I prioritise Recall > F1 > Precision, because the cost of a false negative (missed churner) is higher.

---

### Q25. What would you add to the Streamlit app to make it production-quality?

**Answer**:
1. **Authentication** — Role-based access control for retention team
2. **Batch prediction** — Upload a CSV of customers and download scored results
3. **Explainability panel** — Show the top 3 risk factors for each prediction (SHAP)
4. **Historical tracking** — Store predictions to monitor score changes over time
5. **Feedback loop** — Allow agents to mark actual outcomes to improve the model
6. **API endpoint** — REST API for CRM integration using FastAPI
7. **Monitoring dashboard** — Model drift detection and data quality alerts
