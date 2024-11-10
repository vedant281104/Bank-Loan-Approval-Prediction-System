import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Loan Approval Prediction App")

# Input fields for loan application features
st.header("Applicant Information")
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education Level (0 for Graduate, 1 for Not Graduate)", [0, 1])
self_employed = st.selectbox("Self-Employed (0 for No, 1 for Yes)", [0, 1])
income_annum = st.number_input("Annual Income", min_value=0)

st.header("Loan Details")
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0, step=1)
cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900)

st.header("Assets Information")
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Prepare data for prediction
input_data = np.array([
    no_of_dependents, education, self_employed,
    income_annum, loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value,
    luxury_assets_value, bank_asset_value
]).reshape(1, -1)

# Predict loan status
if st.button("Check Loan Approval Status"):
    prediction = model.predict(input_data)
    st.subheader("Loan Approval Status")
    st.write("Approved" if prediction[0] == 0 else "Rejected")






