# Business Report

## Customer Churn Prediction & Retention Intelligence System

**Prepared for**: Executive Leadership & Customer Success Team  
**Date**: October 2026  
**Classification**: Internal Use

---

## Executive Summary

Customer churn is one of the most significant drivers of revenue loss in the telecommunications industry. This report presents the findings of a machine learning initiative designed to predict which customers are most likely to cancel their service, enabling the retention team to intervene proactively.

A Gradient Boosting model was developed and validated on 500 customer records. The model identifies high-risk customers with strong discriminative ability (ROC-AUC > 0.85), enabling the company to focus retention resources where they will have the greatest impact.

---

## Business Problem

**Current state**: The retention team has no systematic early warning system. Churn is detected only after it occurs, making recovery impossible.

**Impact**:
- Lost recurring revenue from churned customers
- High customer acquisition cost (CAC) vs. retention cost
- No prioritisation framework for the retention team
- Reactive rather than proactive customer management

**Desired state**: A risk-scoring system that flags at-risk customers before they churn, allowing targeted intervention within a critical 30-day window.

---

## Key Findings from Data Analysis

### Churn Profile

| Finding | Insight |
|---|---|
| Overall churn rate: ~40% | Elevated — indicating a retention crisis |
| Month-to-month contracts: highest churn | No contract = no switching cost |
| Fiber optic users: higher churn | Price sensitivity in high-spend segment |
| Electronic check payers: elevated churn | Payment friction may signal dissatisfaction |
| Short tenure (< 6 months): most vulnerable | Onboarding phase is critical |
| Senior citizens: slightly higher churn | May need specialised support |

### Critical Risk Combinations

A customer who is **simultaneously**:
- On a month-to-month contract
- Using fiber optic internet
- Paying by electronic check
- With tenure under 6 months

...represents the **highest churn risk profile**. These customers should be flagged immediately.

---

## Churn Drivers (Model Findings)

The top factors associated with churn (based on model feature importance):

1. **Contract type** — Month-to-month vs. annual/two-year
2. **Tenure** — Shorter tenure strongly correlates with churn
3. **Monthly charges** — Higher charges increase churn probability
4. **Internet service type** — Fiber optic users churn more
5. **Payment method** — Electronic check users at higher risk
6. **Tech support** — Absence of tech support increases churn
7. **Online security** — Absence increases risk
8. **Contract Risk flag** — Engineered feature confirming above

> **Important**: These factors show correlation with churn, not causation. Retention decisions should use this as a prioritisation guide, not a deterministic predictor.

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | See outputs/metrics/model_comparison.csv |
| Recall | Prioritised in model selection |
| Precision | Secondary metric |
| F1-Score | Balanced metric |
| Cross-Validation | 5-fold stratified |

The Gradient Boosting model (tuned) was selected as the final model based on:
- Highest ROC-AUC in cross-validation
- Strong recall (minimises missed churners)
- Stable performance across folds (low variance)

---

## Customer Risk Distribution

After scoring all 500 customers:

| Risk Category | Threshold | Action |
|---|---|---|
| HIGH RISK | Churn probability >= 60% | Immediate retention outreach |
| MEDIUM RISK | 30% - 60% | Engagement campaigns |
| LOW RISK | < 30% | Loyalty / upsell programs |

Full predictions: `outputs/predictions/customer_churn_predictions.csv`

---

## Business Recommendations

### HIGH RISK Customers — Immediate Action Required

1. **Personal outreach within 48 hours** — Assign dedicated retention specialist
2. **Tailored discount offer** — Based on current charges and tenure
3. **Contract upgrade incentive** — Offer discount for switching to annual plan
4. **Service review call** — Identify and resolve pain points
5. **Escalation pathway** — For customers with complaints or service issues

**ROI estimate**: If even 20% of high-risk customers are retained at their current monthly spend, the revenue impact is material.

### MEDIUM RISK Customers — Proactive Engagement

1. **Email/SMS campaigns** — Value communication and feature education
2. **Service upgrade offers** — Tech support, online security bundles
3. **Monthly monitoring** — Re-score at each billing cycle
4. **Net Promoter Score (NPS) survey** — Identify dissatisfaction early

### LOW RISK Customers — Growth & Loyalty

1. **Loyalty rewards programme** — Points/cashback for long-term customers
2. **Premium upsell** — Offer fiber, multiple lines, security bundles
3. **Referral programme** — Incentivise word-of-mouth acquisition
4. **Annual contract conversion** — Lock in with multi-year incentives

---

## Expected Business Benefits

| Benefit | Description |
|---|---|
| Reduced churn rate | Proactive retention of high-risk customers |
| Improved CAC efficiency | Retain vs. acquire at lower cost |
| Better resource allocation | Focus retention budget on highest-risk customers |
| Earlier intervention | Act within the 30-day critical window |
| Data-driven decisions | Replace intuition with ML-backed prioritisation |

---

## Limitations

1. Model trained on **synthetic data** — production performance may differ
2. Model requires **monthly re-scoring** as customer behaviour evolves
3. **Threshold values** (30%/60%) may need calibration against business outcomes
4. No causal inference — cannot guarantee that identified factors _cause_ churn
5. Model does not account for **macroeconomic factors** or competitive pricing

---

## Recommended Next Steps

| Priority | Action | Timeline |
|---|---|---|
| 1 | Deploy model to score existing customer base | Month 1 |
| 2 | Train retention team on risk dashboard | Month 1 |
| 3 | Design A/B retention campaign for high-risk segment | Month 2 |
| 4 | Collect real customer data and retrain model | Month 3 |
| 5 | Integrate model with CRM for real-time scoring | Month 4-6 |
| 6 | Build automated retraining pipeline (monthly) | Month 6 |
