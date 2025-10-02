# app/streamlit_app.py
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("../models/diabetes_model.pkl")

st.title("🩺 AI Health Assistant - Diabetes Prediction")

# Input fields
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict"):
    X = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(X)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")
