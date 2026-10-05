
# Customer Churn Prediction & Retention Intelligence System

> **Week 12 Final Capstone Project | End-to-End Data Science Solution**

## Project Overview

The **Customer Churn Prediction & Retention Intelligence System** is an end-to-end machine learning project designed to identify customers who are at risk of leaving a telecom service.

The project combines data validation, exploratory data analysis, feature engineering, preprocessing, machine learning model development, hyperparameter tuning, model evaluation, customer risk scoring, and business recommendations into one complete pipeline.

The system predicts customer churn probability and categorizes customers into:

- **High Risk**
- **Medium Risk**
- **Low Risk**

The final solution also provides a Streamlit-based dashboard for exploring customer churn risk and supporting retention decisions.

---

## Business Problem

Customer churn directly impacts recurring revenue and long-term customer value.

The business needs to answer questions such as:

- Which customers are most likely to churn?
- What factors are associated with customer churn?
- Which customers should the retention team contact first?
- How can customer risk be converted into actionable retention strategies?
- How can machine learning support proactive customer retention?

The objective is to move from **reactive churn management** to **proactive customer retention**.

> **Customer Acquisition Cost >> Retention Cost**

Therefore, identifying high-risk customers early can help businesses prioritize retention efforts and reduce potential revenue loss.

---

## Objectives

### Business Objectives

- Identify customers with a high probability of churn.
- Segment customers according to churn risk.
- Identify important factors associated with churn.
- Prioritize customers for retention campaigns.
- Generate actionable business recommendations.

### Technical Objectives

- Perform complete data validation and preprocessing.
- Explore customer and churn patterns through EDA.
- Engineer meaningful customer-level features.
- Implement multiple machine learning algorithms.
- Compare model performance using comprehensive evaluation metrics.
- Apply hyperparameter tuning.
- Select the final model using cross-validation.
- Generate customer-level churn probabilities and risk tiers.
- Build an end-to-end reusable ML pipeline.

---

## Dataset

The primary dataset used in this project is the **provided customer churn dataset**.

### Dataset Information

| Property | Value |
|---|---|
| Dataset | Customer Churn Dataset |
| File | `data/raw/customer_churn.csv` |
| Rows | 500 |
| Columns | 9 |
| Churned Customers | 53 |
| Non-Churned Customers | 447 |
| Churn Rate | 10.6% |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Duplicate Customer IDs | 0 |

### Dataset Schema

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Tenure` | Number of months the customer has stayed |
| `MonthlyCharges` | Monthly customer charges |
| `TotalCharges` | Total customer charges |
| `Contract` | Customer contract type |
| `PaymentMethod` | Customer payment method |
| `PaperlessBilling` | Whether paperless billing is enabled |
| `SeniorCitizen` | Senior citizen indicator |
| `Churn` | Target variable indicating whether the customer churned |

### Supporting Datasets

The repository also contains:

- `data/raw/sales_data.csv`
- `data/raw/house_prices.csv`

These datasets are retained as supporting project resources but **are not used in the customer churn model**.

---

## Key Features

- Complete end-to-end customer churn prediction pipeline.
- Data validation and quality checks before modelling.
- Exploratory Data Analysis with multiple visualisations.
- Multiple encoding and preprocessing techniques.
- **8 engineered customer-level features**.
- Train-only threshold calculation for charge-based feature engineering to prevent data leakage.
- Stratified 80/20 train-test split.
- Four machine learning models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Random Forest hyperparameter tuning using `RandomizedSearchCV`.
- 5-fold stratified cross-validation.
- ROC-AUC and PR-AUC evaluation.
- Customer churn probability scoring.
- High, Medium, and Low customer risk classification.
- Top churn-driver analysis using Logistic Regression coefficients.
- Business-oriented retention recommendations.
- Reusable preprocessing and model artefacts.
- Streamlit dashboard for customer risk analysis.
- Automated project testing with **65/65 pytest tests passed**.

---

## Feature Engineering

The project creates **8 engineered features** to improve customer-level churn analysis:

| Feature | Description |
|---|---|
| `AverageMonthlySpend` | Average monthly spending based on customer charges |
| `ChargesPerTenure` | Charges relative to customer tenure |
| `EstimatedCLV` | Estimated customer lifetime value |
| `TenureGroup` | Customer tenure category |
| `ContractRisk` | Risk representation based on contract characteristics |
| `PaymentRisk` | Risk representation based on payment method |
| `HighChargeFlag` | Indicates customers with relatively high charges |
| `CustomerValueSegment` | Customer value categorisation |

Charge-based thresholds are calculated using the **training data only** before being applied to the test data and scoring pipeline. This prevents test-set information from leaking into feature engineering.

---

## Machine Learning Models

Four baseline models were implemented and compared:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Gradient Boosting**

### Hyperparameter Tuning

Random Forest was additionally tuned using:

- `RandomizedSearchCV`
- 20 parameter combinations
- 5-fold Stratified Cross-Validation
- ROC-AUC as the optimization metric

### Best Random Forest Parameters

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
````

Best tuned Random Forest cross-validation ROC-AUC:

```text
0.9914
```

---

## Actual Results

Production model selection is based on **5-fold CV ROC-AUC on training data only**. Held-out test metrics are reported separately and were not used to pick the final model artefact.

### Final Model

**Logistic Regression**

```text
CV ROC-AUC: 0.9943
```

### Model Comparison — Held-Out Test Set

| Model                  | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ---------------------- | -------: | --------: | -----: | -----: | ------: | -----: |
| Random Forest Baseline |     1.00 |    1.0000 | 1.0000 | 1.0000 |   1.000 | 1.0000 |
| Random Forest Tuned    |     0.97 |    0.7857 | 1.0000 | 0.8800 |   1.000 | 1.0000 |
| Gradient Boosting      |     0.99 |    1.0000 | 0.9091 | 0.9524 |   0.999 | 0.9924 |
| Logistic Regression    |     0.97 |    0.7857 | 1.0000 | 0.8800 |   0.998 | 0.9860 |
| Decision Tree          |     0.96 |    0.8182 | 0.8182 | 0.8182 |   0.906 | 0.8217 |

### Cross-Validation ROC-AUC

| Model                  | Mean CV ROC-AUC | Std. Dev. |
| ---------------------- | --------------: | --------: |
| Logistic Regression    |          0.9943 |    0.0031 |
| Random Forest Baseline |          0.9912 |    0.0038 |
| Random Forest Tuned    |          0.9914 |         — |
| Gradient Boosting      |          0.9789 |    0.0327 |
| Decision Tree          |          0.8860 |    0.0749 |

The final model was selected using cross-validation performance on the training data rather than selecting a model based only on the held-out test set.

---

## Customer Risk Distribution

The final model scored all **500 customers** and classified them using the following thresholds:

| Risk Level  | Probability |
| ----------- | ----------- |
| High Risk   | ≥ 50%       |
| Medium Risk | 30% – 50%   |
| Low Risk    | < 30%       |

### Final Distribution

| Risk Level | Customers | Percentage |
| ---------- | --------: | ---------: |
| High       |        70 |      14.0% |
| Medium     |         8 |       1.6% |
| Low        |       422 |      84.4% |
| **Total**  |   **500** |   **100%** |

---

## Top Churn Drivers

The most influential features identified from the final Logistic Regression model were:

| Rank | Feature               | Coefficient |
| ---: | --------------------- | ----------: |
|    1 | `TenureGroup_New`     |      2.9563 |
|    2 | `Tenure`              |      1.7317 |
|    3 | `TenureGroup_Growing` |      1.6847 |
|    4 | `ContractRisk`        |      1.1487 |
|    5 | `EstimatedCLV`        |      1.1468 |

These factors are used to help interpret customer churn risk and support business-oriented retention recommendations.

---

## Business Recommendations

Based on the customer risk scoring system:

### High-Risk Customers

* Prioritize immediate retention outreach.
* Offer personalised retention incentives.
* Review contract and pricing concerns.
* Provide proactive customer support.
* Focus on customers showing early signs of disengagement.

### Medium-Risk Customers

* Place customers into proactive monitoring campaigns.
* Send personalised engagement offers.
* Monitor changes in usage and billing behaviour.
* Encourage longer-term contract adoption where appropriate.

### Low-Risk Customers

* Continue standard engagement.
* Focus on loyalty and upselling opportunities.
* Maintain service quality.
* Monitor customers for changes that could increase churn risk.

---

## Project Architecture

```text
Raw Customer Data
       ↓
Data Validation
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Preprocessing Pipeline
       ↓
Model Training
       ↓
Cross-Validation
       ↓
Hyperparameter Tuning
       ↓
Model Comparison
       ↓
Final Model Selection
       ↓
Customer Churn Probability
       ↓
Risk Classification
       ↓
Business Recommendations
       ↓
Streamlit Dashboard
```

---

## Repository Structure

```text
Customer-Churn-Prediction---Retention-Intelligence-System/
│
├── data/
│   └── raw/
│       ├── customer_churn.csv
│       ├── sales_data.csv
│       └── house_prices.csv
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessing_pipeline.pkl
│   └── feature_thresholds.pkl
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── processed_data/
│   └── reports/
│
├── src/
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── ...
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── ...
│
├── app.py
├── run_project.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chandravanshitassu-lab/Customer-Churn-Prediction---Retention-Intelligence-System.git
```

### 2. Navigate to the Project

```bash
cd Customer-Churn-Prediction---Retention-Intelligence-System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Run the Complete ML Pipeline

```bash
python run_project.py
```

The pipeline performs:

1. Dataset loading
2. Data validation
3. Exploratory Data Analysis
4. Feature engineering
5. Train/test splitting
6. Preprocessing
7. Model training
8. Cross-validation
9. Hyperparameter tuning
10. Model evaluation
11. Final model selection
12. Customer risk scoring
13. Business recommendation generation

### Run Tests

```bash
python -m pytest -v
```

**Latest verified result:**

```text
65/65 tests passed in 5.49 seconds.
```

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Technologies

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Validation

* Cross-validation
* ROC-AUC
* PR-AUC
* Accuracy
* Precision
* Recall
* F1-score

### Hyperparameter Tuning

* RandomizedSearchCV

### Dashboard

* Streamlit

### Testing

* Pytest

### Project Management

* Git
* GitHub

---

## Limitations

1. **Small dataset:** The dataset contains 500 customers, which limits the generalisability of the model. A production system should be validated on a larger and more representative customer population.

2. **Class imbalance:** Churn represents 10.6% of the dataset. Class imbalance is addressed using `class_weight='balanced'` for applicable models, but additional techniques such as SMOTE or cost-sensitive learning could be evaluated.

3. **Model validation:** The reported results are based on a single stratified 80/20 train-test split combined with 5-fold cross-validation on the training data. Additional validation on an independent dataset would provide stronger evidence of production performance.

4. **Static model:** The current system does not include automated model retraining. Customer behaviour and churn patterns may change over time.

5. **Business impact validation:** The system provides retention recommendations, but actual retention uplift and financial ROI would need to be measured through controlled retention campaigns or A/B testing.

---

## Future Improvements

Potential improvements include:

* Increase dataset size using real-world customer records.
* Add additional behavioural and transactional features.
* Evaluate advanced imbalance-handling techniques such as SMOTE.
* Test additional machine learning algorithms.
* Implement automated model retraining.
* Add model monitoring and drift detection.
* Deploy the system to a cloud environment.
* Integrate with CRM systems.
* Add automated retention campaign tracking.
* Measure actual retention uplift and ROI.
* Implement explainable AI techniques such as SHAP.
* Improve dashboard interactivity and business reporting.

---

## Author

**Shivani Shah**

Data Science / Data Analytics Intern

---

## Project Context

This project was developed as part of the **Week 12 Final Capstone Project** for a data science internship.

The project demonstrates an end-to-end machine learning workflow, including:

* Data validation
* Exploratory data analysis
* Feature engineering
* Data preprocessing
* Machine learning
* Hyperparameter tuning
* Cross-validation
* Model evaluation
* Customer risk scoring
* Business recommendations
* Dashboard development
* Automated testing
* Git/GitHub project management

The objective is to demonstrate how machine learning can be transformed from a predictive model into a **business-oriented customer retention intelligence system**.





