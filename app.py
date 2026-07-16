import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
model_columns = joblib.load('models/model_columns.pkl')

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict churn probability.")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
num_services = st.slider("Number of Services Subscribed", 0, 9, 3)

input_dict = {col: 0 for col in model_columns} 

input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges
input_dict['NumServices'] = num_services

if f'Contract_{contract}' in input_dict:
    input_dict[f'Contract_{contract}'] = 1
if f'InternetService_{internet_service}' in input_dict:
    input_dict[f'InternetService_{internet_service}'] = 1
if f'PaymentMethod_{payment_method}' in input_dict:
    input_dict[f'PaymentMethod_{payment_method}'] = 1

input_df = pd.DataFrame([input_dict])

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices']
input_df[num_features] = scaler.transform(input_df[num_features])

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of churn ({probability*100:.1f}% probability)")
    else:
        st.success(f"✅ Low risk of churn ({probability*100:.1f}% probability)")