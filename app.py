import streamlit as st
import pickle
import numpy as np

# Load model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

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

# ---------- ENCODING (SAFE) ----------
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}
property_area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

input_data = np.array([
    gender_map[Gender],
    married_map[Married],
    education_map[Education],
    self_employed_map[Self_Employed],
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    property_area_map[Property_Area]
]).reshape(1, -1)

# ---------- PREDICTION ----------
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
