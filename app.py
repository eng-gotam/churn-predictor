import streamlit as st
import pandas as pd
import joblib

# load the saved model
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Predictor")
st.write("Fill in the customer details below to predict whether they will churn.")

# ── INPUT FIELDS ──────────────────────────────────────────
gender   = st.selectbox("Gender", ["Male", "Female"])
senior   = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
tenure   = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
total    = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)

# ── PREDICT BUTTON ────────────────────────────────────────
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "gender":         gender,
        "SeniorCitizen":  senior,
        "tenure":         tenure,
        "Contract":       contract,
        "MonthlyCharges": monthly,
        "TotalCharges":   total
    }])

    prob = model.predict_proba(input_df)[:, 1][0]
    pred = prob >= 0.35  # custom threshold for catching more churners

    st.markdown("---")
    if pred:
        st.error(f"⚠️ This customer is likely to churn")
        st.metric("Churn Probability", f"{prob:.2%}")
    else:
        st.success(f"✅ This customer is unlikely to churn")
        st.metric("Churn Probability", f"{prob:.2%}")