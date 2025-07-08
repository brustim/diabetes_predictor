import streamlit as st
import pickle
import numpy as np
import os

# Load the model
model_path = os.path.join("model", "diabetes_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Streamlit GUI
st.title("Diabetes Prediction")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=900)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
age = st.number_input("Age", min_value=1, max_value=120)

# Predict button
if st.button("Predict"):
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("⚠️ The model predicts that the person is likely to have diabetes.")
    else:
        st.success("✅ The model predicts that the person is unlikely to have diabetes.")

