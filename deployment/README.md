# Deployment Guide

## Customer Churn Prediction — Streamlit Application

This app scores a **single customer** using the saved model and preprocessing artefacts. It does **not** retrain on startup.

### Schema (actual 9-column dataset)

Inputs match `data/raw/customer_churn.csv` (CustomerID and Churn are not entered):

| Field | Type | Allowed values |
|---|---|---|
| Tenure | numeric | months, ≥ 0 |
| MonthlyCharges | numeric | USD per month, ≥ 0 |
| TotalCharges | numeric | cumulative USD, ≥ 0 |
| Contract | categorical | Month-to-month, One year, Two year |
| PaymentMethod | categorical | Credit Card, Electronic Check, Bank Transfer |
| PaperlessBilling | categorical | Yes, No |
| SeniorCitizen | binary | 0 or 1 |

---

### Prerequisites

```bash
pip install -r requirements.txt
python run_project.py
```

`run_project.py` must have been executed so these files exist:

- `models/best_model.pkl`
- `models/preprocessing_pipeline.pkl`
- `models/feature_thresholds.pkl` (training-only MonthlyCharges cuts)

App-only packages are also listed in `deployment/requirements.txt`: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit.

---

### How to run

```bash
streamlit run deployment/app.py
```

Opens at `http://localhost:8501`.

---

### How it works

1. **Load artefacts** — `deployment/prediction.py` loads the preprocessor, model, and charge thresholds from `models/` (no fitting).
2. **Collect inputs** — the Streamlit form in `deployment/app.py`.
3. **Validate** — `validate_inputs()` checks required numeric fields.
4. **Feature engineering** — same deterministic features as training, then HighChargeFlag / CustomerValueSegment using **saved training cuts**.
5. **Transform** — `preprocessing_pipeline.pkl` (`transform` only).
6. **Predict** — class label from `predict`, probability from `predict_proba`.
7. **Risk level** (from `src/config.py`):
   - High Risk: probability ≥ 0.50
   - Medium Risk: 0.30 ≤ probability < 0.50
   - Low Risk: probability < 0.30
8. **Recommendation** — text tied to the risk band.

---

### Outputs

| Output | Description |
|---|---|
| Prediction | 0 = Will not churn, 1 = Will churn |
| Churn probability | 0.0–1.0 from the saved model |
| Risk level | High Risk / Medium Risk / Low Risk |
| Recommendation | Retention action for that band |

---

### Troubleshooting

| Problem | Solution |
|---|---|
| `FileNotFoundError` for `.pkl` files | Run `python run_project.py` first |
| `ModuleNotFoundError` | Activate venv and `pip install -r requirements.txt` |
| Port already in use | `streamlit run deployment/app.py --server.port 8502` |

---

### Notes

- Predictions come from the saved classifier — they are not hard-coded.
- The app was trained on the provided `data/raw/customer_churn.csv` (500 rows × 9 columns).
- Inference reuses training-only charge thresholds so new customers are not scored with test-set quantiles.
