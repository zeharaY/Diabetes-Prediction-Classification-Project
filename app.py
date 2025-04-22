import streamlit as st
import joblib
import numpy as np

# Load model (we'll create this next)
model = joblib.load('diabetes_predictor.pkl')

st.title("Diabetes Prediction App")
age = st.number_input("Age")
bmi = st.number_input("BMI")
glucose = st.number_input("Fasting Blood Glucose")

if st.button("Predict"):
    features = np.array([age, bmi, glucose]).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
