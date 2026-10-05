"""
deployment/app.py
-----------------
Streamlit web application — Customer Churn Prediction & Retention System.
Uses the actual 9-column dataset schema.

Run with:
    streamlit run deployment/app.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from deployment.prediction import predict_single_customer, validate_inputs

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction & Retention Intelligence System")
st.markdown(
    "> **Week 12 Final Capstone** — Enter customer details to generate a "
    "real-time churn probability score, risk category, and retention recommendation."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("🧾 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Numerical Features**")
    tenure          = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0, max_value=500, value=65)
    total_charges   = st.number_input("Total Charges ($)", min_value=0, max_value=50000,
                                      value=int(tenure * monthly_charges))
    senior_citizen  = st.selectbox("Senior Citizen", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No")

with col2:
    st.markdown("**Categorical Features**")
    contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method   = st.selectbox("Payment Method",
                                    ["Credit Card", "Electronic Check", "Bank Transfer"])
    paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
    inputs = {
        "Tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "Contract":         contract,
        "PaymentMethod":    payment_method,
        "PaperlessBilling": paperless,
        "SeniorCitizen":    senior_citizen,
    }

    errors = validate_inputs(inputs)
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
    else:
        try:
            result = predict_single_customer(inputs)
            prob   = result["probability"]
            risk   = result["risk_level"]
            pred   = result["prediction"]
            rec    = result["recommendation"]

            st.divider()
            st.subheader("📋 Prediction Results")

            icon = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}.get(risk, "⚪")

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction",       "Will Churn" if pred == 1 else "Will Not Churn")
            c2.metric("Churn Probability", f"{prob * 100:.1f}%")
            c3.metric("Risk Level",        f"{icon} {risk}")

            st.markdown(f"**Churn Probability: {prob * 100:.1f}%**")
            st.progress(prob)

            if risk == "High Risk":
                st.error(f"### {icon} {risk}\n\n{rec}")
            elif risk == "Medium Risk":
                st.warning(f"### {icon} {risk}\n\n{rec}")
            else:
                st.success(f"### {icon} {risk}\n\n{rec}")

            with st.expander("📊 View Input Summary"):
                st.json(inputs)

        except FileNotFoundError as e:
            st.error(
                f"❌ Model artefacts not found.\n\n"
                f"Please run the training pipeline first:\n"
                f"```\npython run_project.py\n```\n\nError: {e}"
            )
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Customer Churn Prediction System**

    - **Dataset**: 500 customer records (9 features)
    - **Final Model**: Logistic Regression
    - **Churn Rate**: 10.6% (imbalanced)

    **Risk Thresholds**
    - 🔴 High Risk: prob >= 50%
    - 🟡 Medium Risk: 30% - 50%
    - 🟢 Low Risk: < 30%

    ---
    *Week 12 Final Capstone*
    """)
    st.markdown("## Quick Reference")
    st.code("python run_project.py")
    st.code("python -m pytest -v")
    st.code("streamlit run deployment/app.py")
