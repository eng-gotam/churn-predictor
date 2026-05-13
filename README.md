# 📊 Customer Churn Prediction

A complete end-to-end machine learning project that predicts whether 
a telecom customer will churn (leave the service), built using the 
IBM Telco Customer Churn dataset.

## 🎯 Objective
Identify customers who are likely to cancel their subscription so the 
business can take early action and improve retention.

## 📁 Dataset
- Source: IBM Telco Customer Churn Dataset
- Rows: 7,043 customers
- Target column: Churn (Yes / No)
- Features used: gender, SeniorCitizen, tenure, MonthlyCharges,
  TotalCharges, Contract, InternetService, OnlineSecurity, PaymentMethod

## ⚙️ What was done

### 1. Exploratory Data Analysis (EDA)
- Checked shape, data types, missing values, and duplicates
- Visualized churn distribution using countplot

### 2. Data Cleaning
- Converted TotalCharges from text to numeric
- Handled missing values using SimpleImputer
- Dropped irrelevant columns (customerID, PhoneService, etc.)
- Encoded target column using LabelEncoder

### 3. Preprocessing Pipeline
- Numeric columns → Impute missing values → StandardScaler
- Categorical columns → Impute missing values → OneHotEncoder
- Combined using ColumnTransformer inside a Scikit-learn Pipeline

### 4. Models Trained
| Model               | Accuracy | ROC-AUC | CV Score |
|---------------------|----------|---------|----------|
| Gradient Boosting   | 0.7928   | 0.8440  | 0.7981   |
| Random Forest       | 0.7693   | 0.8003  | 0.7724   |
| SVM                 | 0.7459   | 0.8151  | 0.7474   |
| Logistic Regression | 0.7402   | 0.8381  | 0.7382   |
| Decision Tree       | 0.7360   | 0.6574  | 0.7236   |

### 5. Evaluation
- Accuracy, ROC-AUC, Cross-validation Score
- Confusion Matrix per model
- Classification Report (precision, recall, F1)
- ROC Curve comparison across all models
- Custom prediction threshold (default 0.35) to maximize 
  recall on churners

### 6. Deployment
- Best model saved using joblib
- Interactive web app built with Streamlit
- Users can input customer details and get instant churn prediction
- Adjustable sensitivity threshold slider
- Retention suggestions based on customer profile

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib

## 🚀 How to run locally
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author
Gotam Kumar
BS Artificial Intelligence — SMIU, Karachi
