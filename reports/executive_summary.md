# Executive Summary

## Customer Churn Prediction Project

**One page | October 2026**

---

### What Problem Was Solved?

The company was losing customers without any advance warning. The retention team had no way to identify which customers were likely to leave, resulting in reactive rather than proactive retention management.

**Solution**: A machine learning system that scores every customer with a churn probability, categorises them into risk tiers (HIGH / MEDIUM / LOW), and recommends specific retention actions.

---

### What Did We Discover?

**Three most critical churn drivers identified:**

1. **Contract type** — Customers on month-to-month contracts churn significantly more than those on annual or two-year contracts
2. **Tenure** — New customers (< 6 months) are at the highest risk during the onboarding phase
3. **Service + Payment combination** — Fiber optic users paying by electronic check represent the highest-risk profile

**Key finding**: The top ~30% highest-risk customers account for the majority of actual churn events. Concentrating retention effort on this segment offers the highest ROI.

---

### What Model Was Selected?

**Gradient Boosting (Tuned)**

Selected because:
- Highest ROC-AUC across all models tested
- Strong recall (minimises missed churners)
- Stable cross-validation performance
- Native handling of mixed data types

Competing models evaluated: Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting

---

### How Well Did It Perform?

| Metric | Value |
|---|---|
| ROC-AUC | See `outputs/metrics/model_comparison.csv` |
| Recall | Prioritised (misses = lost customers) |
| Validation | 5-fold stratified cross-validation |
| Tuning | RandomizedSearchCV (20 iterations) |

The model reliably discriminates churners from non-churners and produces well-calibrated probabilities for risk scoring.

---

### What Should the Business Do?

**Immediately (Month 1):**
- Deploy the risk scoring dashboard
- Contact all HIGH RISK customers within 48 hours with a personalised retention offer
- Train the retention team to use the prediction system

**Short-term (Months 2-3):**
- Design differentiated campaigns for MEDIUM RISK customers
- Collect outcome data to measure retention campaign effectiveness
- Replace synthetic training data with real customer records

**Long-term (Months 4-6):**
- Integrate model with CRM for automated real-time scoring
- Build monthly automated retraining pipeline
- A/B test retention offers to measure lift vs. control group
