pip install -r requirements.txt

import streamlit as st
import joblib
import pandas as pd

# Load the saved model and related objects
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# Streamlit app configuration
st.title("Dysphagia Screening Application")
st.write("Enter the following features to predict the value of FILS score.")

# User input fields
fim_motor = st.number_input("FIM motor", min_value=13.0, max_value=91.0, value=13.0, step=1.0)
duration_since_stroke_onset = st.number_input("Duration since stroke onset", min_value=0.0, max_value=365.0, value=0.0, step=1.0)
fim_cognition = st.number_input("FIM cognition", min_value=5.0, max_value=35.0, value=5.0, step=1.0)
japan_coma_scale = st.number_input("Japan Coma Scale", min_value=0.0, max_value=300.0, value=0.0, step=1.0)

# Convert input values to DataFrame
input_data = pd.DataFrame({
    "FIM motor": [fim_motor],
    "Duration since stroke onset": [duration_since_stroke_onset],
    "FIM cognition": [fim_cognition],
    "Japan Coma Scale": [japan_coma_scale]
})

# Scale the data and select features
input_scaled = scaler.transform(input_data)
input_selected = selected_features.transform(input_scaled)

# Execute prediction
if st.button("Predict"):
    prediction = model.predict(input_selected)
    st.subheader("Prediction Result")
    st.write(f"Predicted FILS score: {prediction[0]:.2f}")
