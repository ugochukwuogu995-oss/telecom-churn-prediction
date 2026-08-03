
# Telecom Customer Churn Prediction Using Machine Learning

## Project Overview

Customer churn is one of the biggest challenges faced by telecommunication companies. Losing existing customers leads to reduced revenue and increased customer acquisition costs.

This project develops a Machine Learning model to predict  whether a customer  is likely to churn based on customer demographics, account information, subscribed services, and billing details. The model helps telecom companies identify high-risk customers early and implement effective retention strategies.

## Problem Statement

Telecommunication companies experience significant financial losses due to customer churn. Traditional methods often identify churn only after customers have already left.

This project provides a predictive solution that enables companies to identify customers likely to churn before they leave, allowing proactive intervention.

## Project Objectives

- Collect and preprocess telecom customer data.
- Perform Exploratory Data Analysis (EDA).
- Build multiple Machine Learning classification models.
- Compare model performance.
- Select the best-performing model.
- Deploy the model using Streamlit.
- Provide business recommendations for reducing churn.

 ## Dataset

*Dataset:* IBM Telco Customer Churn Dataset

The dataset contains information about telecom customers including:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Customer Churn

## Target Variable

Churn (Yes / No)

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook
- Git & GitHub

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Selection
8. Model Deployment
9. Business Recommendations


## Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The Random Forest model achieved the best overall performance and was selected as the final prediction model.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## Key Findings

The analysis revealed that customer churn is strongly influenced by:

- Contract Type
- Customer Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Payment Method

Customers with month-to-month contracts and shorter tenure were more likely to churn.

## Business Recommendations

- Introduce loyalty programs for long-term customers.
- Encourage migration to longer contract plans.
- Offer personalized retention packages for high-risk customers.
- Improve customer service quality.
- Monitor high-risk customers using predictive analytics.


## Project Structure


Telecom-Churn-Prediction/
│
├── Telecom_Churn_Prediction.ipynb
├── app.py
├── telecom_churn_model.pkl
├── requirements.txt
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images
├── presentation
└── report

## Installation

Clone the repository:

bash
git clone https://github.com/yourusername/telecom-churn-prediction.git


Move into the project directory:

bash
cd telecom-churn-prediction


Install dependencies:

bash
pip install -r requirements.txt


Run the application:

bash
streamlit run app.py


##  Live Demo

Streamlit App:

(Add your deployed Streamlit link here.)


## Future Improvements

- Hyperparameter tuning
- Explainable AI using SHAP
- Cloud deployment
- Real-time prediction using live telecom data
- Integration with CRM systems

Capstone Project

Telecom Customer Churn Prediction Using Machine Learning


## License

This project is for educational and portfolio purposes.

## Author

Ugochukwu  Monday Ogu
ID FE/25/7566333919
NEXTGEN COHORT
LAGOS
