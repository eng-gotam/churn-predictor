import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── LOAD MODEL ────────────────────────────────────────────────
model = joblib.load("churn_model.pkl")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Predictor")
st.write("Fill in the customer details below to predict whether they will churn.")
st.markdown("---")

# ── INPUT FIELDS ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    tenure     = st.slider("Tenure (months)", 0, 72, 12)
    monthly    = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total      = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)

with col2:
    internet   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    contract   = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment    = st.selectbox("Payment Method", [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                 ])

st.markdown("---")

# ── THRESHOLD SLIDER ──────────────────────────────────────────
threshold = st.slider(
    "Sensitivity threshold (lower = catch more churners)",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)

# ── PREDICT ───────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):

    # exact same column order as X_train
    input_df = pd.DataFrame([{
        "gender":          gender,
        "SeniorCitizen":   senior,
        "tenure":          tenure,
        "InternetService": internet,
        "OnlineSecurity":  online_sec,
        "Contract":        contract,
        "PaymentMethod":   payment,
        "MonthlyCharges":  monthly,
        "TotalCharges":    total
    }])

    # reorder columns to exactly match training
    input_df = input_df[['gender', 'SeniorCitizen', 'tenure', 'InternetService',
                          'OnlineSecurity', 'Contract', 'PaymentMethod',
                          'MonthlyCharges', 'TotalCharges']]

    prob = model.predict_proba(input_df)[:, 1][0]
    pred = prob >= threshold

    st.markdown("### Prediction Result")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Churn Probability", f"{prob:.2%}")
    with col_b:
        st.metric("Threshold Used", f"{threshold:.2%}")

    if pred:
        st.error("⚠️ This customer is LIKELY to churn")
    else:
        st.success("✅ This customer is UNLIKELY to churn")

    # ── PROBABILITY BAR ───────────────────────────────────────
    st.markdown("#### Churn risk level")
    st.progress(float(prob))

    # ── RETENTION TIPS ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Retention suggestions")
    if internet == "Fiber optic" and contract == "Month-to-month":
        st.warning("📌 Fiber optic + month-to-month is highest churn combo — offer a contract upgrade.")
    if tenure < 12:
        st.warning("📌 Customer is in first year — offer a loyalty discount.")
    if online_sec == "No":
        st.info("📌 No online security — offering it may improve retention.")
    if not pred:
        st.info("📌 Customer looks stable — no immediate action needed.")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About this app")
    st.write("Trained on IBM Telco Customer Churn dataset.")
    st.markdown("**Models trained:**")
    st.write("- Logistic Regression")
    st.write("- Decision Tree")
    st.write("- Random Forest")
    st.write("- Gradient Boosting")
    st.write("- SVM")
    st.markdown("**Best model** auto-selected by accuracy.")
    st.markdown("---")
    st.markdown("**Columns used:**")
    st.code("""gender, SeniorCitizen, tenure,
InternetService, OnlineSecurity,
Contract, PaymentMethod,
MonthlyCharges, TotalCharges""")
    st.markdown("---")
    st.caption("Built by Gotam Kumar — SMIU AI Department")
