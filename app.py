import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and encoder
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.set_page_config(page_title="Loan Prediction System")

st.title("🏦 Loan Prediction System")
st.write("Predict whether a loan will be approved or not")

# ---------- USER INPUTS ----------
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

# ---------- ENCODING ----------
def encode(val):
    return le.fit_transform([val])[0]

input_data = np.array([
    encode(Gender),
    encode(Married),
    encode(Education),
    encode(Self_Employed),
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    encode(Property_Area)
]).reshape(1, -1)

# ---------- PREDICTION ----------
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
