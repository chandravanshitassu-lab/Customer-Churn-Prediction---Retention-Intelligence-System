# Speaker Notes
## Customer Churn Prediction & Retention Intelligence System

---

## Slide 1 — Title
"Good [morning/afternoon], everyone. My name is [Your Name], and today I'm presenting my Week 12 Final Capstone Project: a Customer Churn Prediction and Retention Intelligence System.

This is a complete, professional-grade data science project built using real customer data. It covers the full data science lifecycle — from raw data ingestion all the way through to a live web application that any retention team member can use to score a customer in real time.

Let me start by explaining the business problem I set out to solve."

---

## Slide 2 — Business Problem
"The company we're working with faces a common but costly challenge: customers are leaving — churning — without any early warning.

Right now, the retention team only finds out a customer has churned after it's already happened. At that point, it's too late to do anything. There's no early alert, no prioritisation list, no data-driven way to know which customers to call first.

With a 10.6% churn rate, that's 53 customers in this dataset who left. Each of those represents lost recurring revenue — and because it costs significantly more to acquire a new customer than to retain an existing one, this is a serious business problem.

My solution: predict churn BEFORE it happens."

---

## Slide 3 — Objectives
"I set two layers of objectives for this project.

On the business side: I want to score every customer with a probability of churning, rank them by risk, and give the retention team clear, actionable recommendations — not just a list of numbers.

On the technical side: I want a reproducible, tested pipeline. When I say tested, I mean a pytest test suite with 50 automated tests — all of which pass.

My success metrics were ambitious: ROC-AUC above 0.90 and Recall above 0.85. I'll show you in a few slides that I exceeded both."

---

## Slide 4 — Dataset
"The dataset has 500 rows and 9 columns. It's a clean dataset — no missing values, no duplicates.

The target column is Churn: 1 if the customer left, 0 if they stayed. The churn rate is 10.6%, meaning 53 out of 500 customers churned. This is a class imbalance problem — more on how I handled that in a moment.

The features cover four dimensions of the customer relationship: how long they've been a customer (Tenure), what they pay (MonthlyCharges, TotalCharges), how they pay (PaymentMethod, PaperlessBilling), and what type of commitment they've made (Contract).

I also validated the data programmatically — running 24 automated checks to confirm the dataset meets quality standards. Everything passed."

---

## Slide 5 — EDA
"Let me walk you through the most important findings from my exploratory data analysis.

I generated 13 professional visualisations. Here are the key takeaways:

First, the dataset is imbalanced: 89% of customers did NOT churn. This means accuracy alone is a misleading metric — a model that predicts 'no churn' for everyone gets 89% accuracy but is completely useless.

Second, tenure is a clear separator. Churned customers have much lower average tenure. New customers in their first year are at the highest risk.

Third, monthly charges show price sensitivity. Customers who churn tend to be paying more per month."

---

## Slide 6 — Key Insights
"Let me highlight the three most actionable insights from the EDA.

One: Contract type is the most powerful categorical predictor. Month-to-month customers have a noticeably higher churn rate than customers on one-year or two-year contracts. This makes business sense — they have zero switching cost.

Two: Payment method matters. Electronic Check users show elevated churn. This could indicate payment friction, financial instability, or lower customer satisfaction.

Three: The first 12 months are the critical window. New customers are at the highest risk. If we can improve onboarding and retention in the first year, we significantly reduce overall churn.

These insights directly shaped the retention recommendations I'll show you later."

---

## Slide 7 — Feature Engineering
"Beyond the 9 original columns, I engineered 8 additional features — all with clear business rationale.

For example: ChargesPerTenure divides MonthlyCharges by Tenure to capture relative cost pressure. A customer paying $100/month for 2 months is under very different financial pressure than one paying $100/month for 5 years.

EstimatedCLV — Customer Lifetime Value — captures the total revenue the company has generated from that customer. High-CLV customers are the highest retention priority.

ContractRisk and PaymentRisk are binary flags that directly encode the most important categorical signals into features the model can use efficiently.

Crucially, all features were engineered BEFORE the train-test split. The preprocessing pipeline was fitted ONLY on training data to prevent data leakage."

---

## Slide 8 — Machine Learning Models
"I trained four classification algorithms, each chosen for a specific reason.

Logistic Regression is my linear baseline. It's interpretable via coefficients and fast to train.

Decision Tree is a non-linear baseline — produces human-readable rules but prone to overfitting.

Random Forest is my primary ensemble model — robust, handles mixed features well.

Gradient Boosting is the sequential correction approach — typically achieves high accuracy.

An important note: with 10.6% minority class, I used class_weight='balanced' for all models except Gradient Boosting. Without this, models would ignore the churn class entirely and just predict 'no churn' for everyone.

For hyperparameter tuning, I used RandomizedSearchCV with 20 iterations and 5-fold stratified cross-validation."

---

## Slide 9 — Model Evaluation
"Here are the actual results from running the pipeline.

Random Forest achieved perfect scores on the test set — 100% accuracy, 100% recall, 1.0 ROC-AUC. I want to be transparent about this: on a 500-row dataset with only 53 positive examples, perfect test accuracy can indicate the model has memorised the patterns rather than generalised.

That's why cross-validation is the more reliable estimate. The 5-fold CV ROC-AUC for Random Forest is 0.9918, with a very low standard deviation of 0.0033. This means the model is genuinely strong and stable, not just lucky on one test split.

Logistic Regression's CV AUC of 0.9943 is also excellent — and arguably more trustworthy because it's a simpler model.

On why I prioritise Recall over Accuracy: missing a churner — false negative — means the customer leaves and revenue is lost permanently. A false positive just means a wasted retention call, which is recoverable."

---

## Slide 10 — Best Model & Interpretation
"I selected Random Forest (Tuned) as the final model based on its CV AUC of 0.9914 and perfect recall.

The most important finding from the feature importance analysis: Tenure and tenure-derived features account for more than 60% of the model's predictive power. This validates our EDA insight — new customers are at the highest risk.

ChargesPerTenure and EstimatedCLV — both engineered features — also rank highly, which validates that the feature engineering was valuable.

ContractRisk appearing in the top 10 confirms that contract type is a key churn signal, consistent with what we saw in EDA.

One important caveat: feature importance shows correlation with churn, NOT causation. We can't say that high charges CAUSE churn — only that they're associated with it in this dataset."

---

## Slide 11 — Risk Segmentation
"After training the model, I scored all 500 customers.

67 customers — 13.4% — are flagged as HIGH RISK with churn probability of 50% or more. These need immediate retention action.

9 customers are MEDIUM RISK — a monitoring and engagement list.

424 customers are LOW RISK — stable base for loyalty and upsell programmes.

The output file, customer_risk_predictions.csv, contains every customer's ID, probability, prediction, and risk level. This is what the retention team receives.

If we estimate average monthly charges of around $113 for high-risk customers, and we retain even half of them, we protect approximately $3,785 in monthly recurring revenue."

---

## Slide 12 — Business Recommendations
"Let me now connect the model outputs to concrete business actions.

For HIGH RISK customers: the retention team should call within 48 hours. The offer should be personalised based on the specific risk factors. For month-to-month customers, offer a discount for switching to an annual plan. For Electronic Check users, offer easy switching to auto-payment with a small incentive.

For MEDIUM RISK customers: enrol in automated email/SMS campaigns. Communicate product value. Re-score at the next billing cycle.

For LOW RISK customers: focus on upsell and loyalty. These customers are stable and represent growth opportunities.

At the structural level: I recommend redesigning the onboarding programme for new customers in their first 12 months, and reviewing the Electronic Check payment experience."

---

## Slide 13 — Deployment
"The model doesn't just live in a Jupyter notebook — I deployed it as a real web application.

The Streamlit app loads the saved model and preprocessing pipeline from disk. It never retrains. A user enters 7 customer attributes — 4 numeric and 3 categorical — and in real time receives a churn probability, risk level, and personalised retention recommendation.

The architecture is clean: user input goes through validation, then feature engineering (the same features calculated at training time), then the preprocessing pipeline, then the model, then post-processing to produce the risk classification.

To run it: streamlit run deployment/app.py. It opens immediately at localhost:8501."

---

## Slide 14 — Testing & Quality
"A professional project needs professional testing.

I wrote 50 automated pytest tests covering every component of the system:
- 10 data tests: file existence, shape, columns, target validity
- 10 deployment tests: prediction function, input validation
- 15 feature tests: all 8 engineered features, binary flags, categories
- 7 model tests: training, predictions, probabilities, saved artefacts
- 8 preprocessing tests: pipeline output, no NaN, stratification

All 50 passed in 7.49 seconds.

The project also includes technical documentation, a business report, an executive summary, a data dictionary, and 25 interview Q&As — this is a portfolio-ready, presentation-ready capstone."

---

## Slide 15 — Conclusion
"To summarise: I built a complete customer churn prediction system that goes from raw data to live deployment.

The final model — Random Forest Tuned — achieves a cross-validation ROC-AUC of 0.9914 with near-zero variance, meaning it generalises reliably. It identified 67 high-risk customers who should be contacted immediately.

The most important business insight is that TENURE is the primary churn driver. New customers in their first 12 months need the most attention.

Looking ahead, the most impactful improvements would be: collecting more real production data, adding SHAP values for individual-level explanation, and integrating with the CRM via a REST API for automated real-time scoring.

Thank you. I'm happy to take questions."
