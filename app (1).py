
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model
# -------------------------------
from pathlib import Path

MODEL_PATH = Path(_file_).parent / "churn_model.pkl"
model = joblib.load(MODEL_PATH)
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Telecom Customer Churn Prediction")

st.markdown("
Predict whether a telecom customer is likely to leave the company using Machine Learning.

This project was developed as a Capstone Project using the IBM Telco Customer Churn Dataset.
")

# Sidebar
st.sidebar.title("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    0.0,
    200.0,
    70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

# ---------------------------------
# Encode Inputs
# ---------------------------------

gender = 0 if gender == "Female" else 1

partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
phone_service = 1 if phone_service == "Yes" else 0
paperless = 1 if paperless == "Yes" else 0

multiple_lines_map = {
    "No": 0,
    "Yes": 1,
    "No phone service": 2
}

internet_service_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

service_map = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

# ---------------------------------
# Prediction
# ---------------------------------

if st.button("Predict Customer Churn"):

    customer = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines_map[multiple_lines]],
        "InternetService":[internet_service_map[internet_service]],
        "OnlineSecurity":[service_map[online_security]],
        "OnlineBackup":[service_map[online_backup]],
        "DeviceProtection":[service_map[device_protection]],
        "TechSupport":[service_map[tech_support]],
        "StreamingTV":[service_map[streaming_tv]],
        "StreamingMovies":[service_map[streaming_movies]],
        "Contract":[contract_map[contract]],
        "PaperlessBilling":[paperless],
        "PaymentMethod":[payment_map[payment]],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total]
    })

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0]

    churn_probability = probability[1] * 100

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Customer is likely to churn.")

    else:

        st.success("✅ Customer is likely to remain.")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2f}%"
    )

    if churn_probability >= 70:
        st.warning("Risk Level: HIGH")

    elif churn_probability >= 40:
        st.info("Risk Level: MEDIUM")

    else:
        st.success("Risk Level: LOW")

    st.subheader("Business Recommendation")

    if prediction == 1:

        st.write("
- Contact the customer immediately.
- Offer a loyalty discount.
- Recommend a longer-term contract.
- Improve customer support.
- Follow up within the next 30 days.
")

    else:

        st.write("
- Continue providing quality service.
- Offer loyalty rewards.
- Encourage long-term subscriptions.
- Maintain regular customer engagement.
")

# ---------------------------------
# Dashboard
# ---------------------------------

st.markdown("---")

st.subheader("📊 Project Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "7,043")

with col2:
    st.metric("Features", "19")

with col3:
    st.metric("Model", "Random Forest")

st.markdown("---")

st.subheader("📌 Key Factors Affecting Customer Churn")

st.write("
The model identified these features as the most important predictors of customer churn:

- Contract Type
- Customer Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Payment Method
")

st.markdown("---")

st.subheader("💼 Business Recommendations")

st.write("
Telecommunication companies can reduce customer churn by:

- Offering loyalty rewards to long-term customers.
- Encouraging customers to switch from month-to-month contracts to yearly contracts.
- Improving customer service and technical support.
- Providing personalized discounts for high-risk customers.
- Using predictive analytics to identify customers likely to churn.
")

st.markdown("---")

st.subheader("ℹ️ About This Project")

st.write("
*Project Title:* Telecom Customer Churn Prediction Using Machine Learning

*Author:* Ugochukwu Monday Ogu
FE/25/7566333919
NEXTGEN COHORT
LAGOS STATE

*Program:* School of Data & AI Capstone Project

*Technologies Used:*
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

*Dataset:* IBM Telco Customer Churn Dataset
")

st.markdown("---")
st.caption("© 2026 Ugochukwu Ogu | Telecom Customer Churn Prediction")
