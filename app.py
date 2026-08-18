
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="Telecom Churn Model",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Telecom Churn Model")
st.markdown("""
**Author:** Ogu Ugochukwu Monday  
**3MTT Fellow ID:** FE/25/7566333919  
**Cohort:** NextGen Cohort Lagos
""")

st.info("""
**Educational Purpose:**  
This project is designed for educational purposes as part of a 3MTT capstone project.
It demonstrates the use of machine learning to predict customer churn and should not
be used as the sole basis for real-world business decisions.
""")
st.markdown("### 📌 About the Project")

st.write("""
The Telecom Churn Model is a machine learning project developed to identify
telecom customers who may be at risk of leaving a service provider.

The model analyses customer characteristics such as tenure, contract type,
internet service, payment method, monthly charges and other service-related
information to estimate the likelihood of churn.

The goal is to help telecom businesses identify high-risk customers early
and support proactive customer retention strategies.
""")

# ============================================================
# NIGERIAN TELECOM CONTEXT
# ============================================================

st.markdown("### 🇳🇬 Nigerian Telecom Context")

st.write("""
Telecommunications plays an important role in connecting individuals,
businesses and communities across Nigeria. Customer retention is therefore
an important consideration for telecom service providers.

In the Nigerian context, customers may change or discontinue telecom
services because of factors such as affordability, service experience,
network performance, contract preferences, payment options and changing
customer needs.

This project demonstrates how customer data and machine learning can be
used to identify customers who may be at risk of churn. The predictions
can support telecom businesses in prioritising customer engagement and
developing appropriate retention strategies.

For example, a provider could use higher-risk predictions to consider
personalised offers, customer-service follow-ups, suitable subscription
plans, loyalty initiatives and investigation of service-related concerns.

The model is intended as a decision-support tool and should be combined
with business knowledge and customer feedback when making real-world
decisions.
""")
# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("🔬 Project Methodology"):

    st.markdown("""
    ### 1. Data Preparation

    The telecom customer dataset is loaded and cleaned before modelling.
    Customer ID fields are removed because they do not provide useful
    predictive information.

    Missing numerical values are handled using median imputation, while
    missing categorical values are handled using the most frequent value.

    ### 2. Feature Engineering

    Numerical variables are standardised using `StandardScaler`.

    Categorical variables are converted into numerical representations
    using `OneHotEncoder`.

    ### 3. Model Development

    A **Logistic Regression** algorithm is used to predict whether a
    customer is likely to churn.

    The dataset is divided into:

    - **80% training data**
    - **20% testing data**

    Stratified sampling is used to maintain the proportion of churned
    and retained customers in both datasets.

    ### 4. Model Evaluation

    The model is evaluated using:

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Confusion Matrix

    ### 5. Business Application

    The prediction can help telecom businesses identify customers who
    may be at higher risk of leaving and support proactive customer
    retention strategies.
    """)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    possible_files = [
        "Telecom Churn Model1.csv",
        "telecom_churn_model1.csv",
        "Telecom_Churn_Model1.csv"
    ]

    for file in possible_files:
        try:
            return pd.read_csv(file)
        except FileNotFoundError:
            continue

    return None


df = load_data()


if df is None:

    st.error(
        "Dataset not found. Please make sure "
        "'Telecom Churn Model1.csv' is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df.columns = df.columns.str.strip()

# Remove unnecessary customer ID column if present
id_columns = [
    "customerID",
    "CustomerID",
    "customer_id"
]

for col in id_columns:
    if col in df.columns:
        df = df.drop(columns=[col])


# Clean TotalCharges if it exists
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


# ============================================================
# TARGET VARIABLE
# ============================================================

target = "Churn"

if target not in df.columns:

    st.error(
        f"The dataset does not contain the expected target column '{target}'."
    )

    st.write("Columns found in dataset:")
    st.write(df.columns.tolist())

    st.stop()


# Convert target to numeric
df[target] = df[target].map({
    "Yes": 1,
    "No": 0,
    1: 1,
    0: 0
})


df = df.dropna(subset=[target])


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target])
y = df[target]


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN MODEL
# ============================================================

with st.spinner("Training Logistic Regression model..."):

    model.fit(
        X_train,
        y_train
    )

# ============================================================
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("👤 Customer Information")


# Helper function
def get_options(column):

    if column in df.columns:
        return sorted(
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    return []


# Create customer input dictionary
customer = {}
st.markdown("### 📊 Churn Overview")

churn_rate = df["Churn"].mean() * 100
total_customers = len(df)
churned_customers = int(df["Churn"].sum())
retained_customers = total_customers - churned_customers

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

with col3:
    st.metric(
        "Overall Churn Rate",
        f"{churn_rate:.1f}%"
    )
st.markdown("#### Customer Churn Distribution")

churn_chart = df["Churn"].value_counts()

chart_data = pd.DataFrame({
    "Customer Status": ["Stayed", "Churned"],
    "Customers": [
        churn_chart.get(0, 0),
        churn_chart.get(1, 0)
    ]
})

st.bar_chart(
    chart_data.set_index("Customer Status")
)
# ============================================================
# CUSTOMER INPUTS
# ============================================================

for column in X.columns:

    if column in numeric_features:

        minimum = float(df[column].min())
        maximum = float(df[column].max())
        default = float(df[column].median())

        customer[column] = st.sidebar.number_input(
            column,
            min_value=minimum,
            max_value=maximum,
            value=default
        )

    else:

        options = get_options(column)

        if options:

            customer[column] = st.sidebar.selectbox(
                column,
                options
            )
# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

if st.button(
    "🔮 Predict Customer Churn",
    type="primary"
):

    customer_df = pd.DataFrame(
        [customer]
    )

    # Make prediction
    prediction = model.predict(
        customer_df
    )[0]

    # Get churn probability
    probability = model.predict_proba(
        customer_df
    )[0][1]

    churn_probability = probability * 100

    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ This customer is likely to churn."
        )

    else:

        st.success(
            "✅ This customer is likely to remain."
        )

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2f}%"
    )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if churn_probability >= 70:

        st.warning(
            "Risk Level: HIGH"
        )

    elif churn_probability >= 40:

        st.info(
            "Risk Level: MEDIUM"
        )

    else:

        st.success(
            "Risk Level: LOW"
        )

    # ========================================================
    # BUSINESS RECOMMENDATION
    # ========================================================

    st.subheader("💡 Recommended Retention Actions")

    if prediction == 1:

        st.warning(
            "⚠️ Priority: This customer should be considered for retention action."
        )

        st.markdown("""
        **Recommended actions for this high-risk customer:**

        - **Personalised retention offer:** Consider targeted data,
          voice or subscription incentives.
        - **Customer service follow-up:** Contact the customer to
          identify and resolve service complaints.
        - **Affordable plan options:** Consider a plan that better
          matches the customer's needs and budget.
        - **Long-term contract incentive:** Offer suitable benefits
          for moving to a longer-term plan.
        - **Proactive engagement:** Monitor this customer closely
          because of the elevated churn probability.
        - **Service quality review:** Investigate network or service
          quality issues where relevant.
        """)

    else:

        st.success(
            "✅ This customer is currently at lower churn risk."
        )

        st.markdown("""
        **Recommended actions for this customer:**

        - Maintain good service quality.
        - Continue regular customer engagement.
        - Consider appropriate loyalty rewards.
        - Monitor changes in customer behaviour.
        - Encourage continued long-term customer relationships.
        """)
   
# ============================================================
# PROJECT DASHBOARD
# ============================================================

st.markdown("---")

st.subheader("📊 Project Dashboard")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Dataset Size",
        f"{len(df):,}"
    )


with col2:
    st.metric(
        "Features",
        len(X.columns)
    )


with col3:
    st.metric(
        "Model",
        "Logistic Regression"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader("📈 Model Performance")

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

with metric1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with metric2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with metric3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with metric4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

with metric5:
    st.metric(
        "ROC-AUC",
        f"{roc_auc * 100:.2f}%"
    )
# ============================================================
# MODEL VISUALIZATIONS
# ============================================================

st.markdown("---")

st.subheader("📊 Model Evaluation Visualizations")

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

st.markdown("### 🔲 Confusion Matrix")

fig_cm, ax_cm = plt.subplots()

ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=["Stayed", "Churned"]
).plot(ax=ax_cm)

ax_cm.set_title("Confusion Matrix")

st.pyplot(fig_cm)

plt.close(fig_cm)


# ------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------

st.markdown("### 📈 ROC Curve")

fig_roc, ax_roc = plt.subplots()

RocCurveDisplay.from_predictions(
    y_test,
    y_probability,
    ax=ax_roc
)

ax_roc.set_title(
    f"ROC Curve (AUC = {roc_auc:.3f})"
)

st.pyplot(fig_roc)

plt.close(fig_roc)

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔲 Confusion Matrix")

confusion_data = pd.DataFrame(
    conf_matrix,
    index=["Actual Stayed", "Actual Churned"],
    columns=["Predicted Stayed", "Predicted Churned"]
)

st.dataframe(
    confusion_data,
    use_container_width=True
)
st.subheader("💼 Business Insights")

st.markdown("""
### 🇳🇬 Application to the Nigerian Telecom Market

For a Nigerian telecom service provider, churn predictions could support
a more proactive customer-retention strategy.

**Potential applications include:**

- **Affordability:** Identify customers who may benefit from more suitable
  subscription or pricing options.
- **Customer service:** Prioritise follow-up with customers showing
  elevated churn risk.
- **Service experience:** Investigate possible service-related concerns
  among high-risk customers.
- **Personalised engagement:** Develop targeted offers based on customer
  characteristics and service usage.
- **Customer loyalty:** Encourage continued relationships through
  appropriate loyalty initiatives.
- **Retention prioritisation:** Focus limited retention resources on
  customers with higher predicted churn risk.

The model does not determine why an individual customer will churn.
Instead, it provides a risk signal that can help businesses decide
which customers may require further investigation or engagement.
""")
# ============================================================
# CONCLUSION
# ============================================================

st.markdown("---")

st.subheader("🏁 Conclusion")

st.markdown("""
This Telecom Churn Prediction Model demonstrates how machine learning
can be used to identify customers who may be at risk of leaving a
telecom service provider.

The Logistic Regression model achieved an accuracy of **80.55%** and
a ROC-AUC score of **84.19%**, demonstrating good overall predictive
performance.

The application goes beyond prediction by providing churn
probabilities, customer risk levels, and recommended retention actions.

These insights can help telecom businesses identify high-risk
customers and support proactive customer retention strategies.

**Note:** This application is designed for educational purposes and
should be used as a decision-support tool rather than the sole basis
for real-world business decisions.
""")
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "© 2026 Ogu Ugochukwu Monday. "
    "This project is designed for Educational purposes."
)
