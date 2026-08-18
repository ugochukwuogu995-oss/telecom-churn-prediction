
# Telecom Churn Prediction Model

## 📱 Project Overview

The **Telecom Churn Prediction Model** is a machine learning application developed to predict whether a telecom customer is likely to leave a service provider.

The project uses customer information such as tenure, contract type, internet service, payment method, monthly charges, and other service-related characteristics to estimate the probability of customer churn.

The application was developed as part of a **3MTT Capstone Project**.

## 🇳🇬 Nigerian Telecom Context

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

For example, a telecom provider could use higher-risk predictions to
consider personalised offers, customer-service follow-ups, suitable
subscription plans, loyalty initiatives and investigation of
service-related concerns.

The model is designed as a decision-support tool and should be combined
with business knowledge and customer feedback when making real-world
decisions.
## 👤 Author

**Name:** Ogu Ugochukwu Monday  
**3MTT Fellow ID:** FE/25/7566333919  
**Cohort:** NextGen Cohort Lagos

## 🎯 Project Objective

The main objective of this project is to develop a machine learning model that can:

- Predict whether a customer is likely to churn.
- Estimate the probability of customer churn.
- Classify customers into low, medium, and high-risk categories.
- Provide recommended customer retention actions.
- Demonstrate how machine learning can support telecom business decisions.

## 🗂️ Dataset

The project uses a telecom customer churn dataset containing customer demographic, account, service, and billing information.

The target variable is:

- `Churn = Yes` → Customer churned
- `Churn = No` → Customer remained

Customer identification fields such as `customerID` are removed because they do not provide useful predictive information.

## 🔧 Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## 🤖 Machine Learning Approach

The project uses **Logistic Regression** for binary classification.

### Data preprocessing

The preprocessing pipeline includes:

1. Numerical missing-value imputation using the median.
2. Categorical missing-value imputation using the most frequent value.
3. Numerical feature scaling using `StandardScaler`.
4. Categorical feature encoding using `OneHotEncoder`.

A `ColumnTransformer` and Scikit-learn `Pipeline` are used to combine the preprocessing and machine-learning steps.

## 📊 Train/Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

Stratified sampling is used to preserve the distribution of churn and non-churn customers.

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## 📊 Model Results

The Logistic Regression model was evaluated on the test dataset using multiple performance metrics.

| Metric | Result |
|---|---:|
| Customer Churn Rate | 26.5% |
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1 Score | 60.40% |
| ROC-AUC | 84.19% |

The model achieved an accuracy of **80.55%** and an ROC-AUC of **84.19%**, demonstrating good overall ability to distinguish between customers who are likely to churn and those who are likely to remain.

The precision score of **65.72%** indicates that a substantial proportion of customers predicted as churners were actual churners. The recall score of **55.88%** indicates that the model identified more than half of the customers who actually churned.

The F1 score of **60.40%** reflects a moderate balance between precision and recall.

Although the results are promising, the recall score indicates that further model improvement could help identify more customers at risk of churn.
## 📊 Application Features

The Streamlit application provides:

### Customer Churn Overview

Displays:

- Total customers
- Number of churned customers
- Overall churn rate
- Customer churn distribution

### Customer Prediction

Users can enter customer information through the sidebar and generate:

- Churn prediction
- Churn probability
- Risk classification

### Risk Classification

Customers are classified as:

- **LOW RISK:** below 40%
- **MEDIUM RISK:** 40%–69.99%
- **HIGH RISK:** 70% and above

### Retention Recommendations

The application provides suggested actions for customers identified as having elevated churn risk.

## ("💼 Business Insights")

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

## 🚀 How to Run the Application

Install the required packages:

```bash
pip install streamlit pandas scikit-learn matplotlib

⚠️ Disclaimer

This project is developed for educational purposes as part of a 3MTT capstone project.

The predictions generated by the model should not be used as the sole basis for real-world customer or business decisions.

🏁 Conclusion

This project demonstrates how machine learning can be applied to telecom customer data to predict churn and support proactive customer retention.

The combination of data preprocessing, Logistic Regression, model evaluation, probability-based risk classification, and business recommendations provides an end-to-end example of a practical machine-learning application.
