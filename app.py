import streamlit as st
import joblib
import pandas as pd

# Load the saved model and related objects
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app configuration
st.title("Dysphagia Screening Application")
st.write("Enter the following features to predict the value of the FILS score.")

# User input fields
fim_motor = st.number_input("FIM Motor", min_value=13.0, max_value=91.0, value=13.0, step=1.0)
duration_since_stroke_onset = st.number_input("Duration Since Stroke Onset (days)", min_value=0.0, max_value=365.0, value=0.0, step=1.0)
fim_cognition = st.number_input("FIM Cognition", min_value=5.0, max_value=35.0, value=5.0, step=1.0)
japan_coma_scale = st.number_input("Japan Coma Scale", min_value=0.0, max_value=300.0, value=0.0, step=1.0)

# Convert input values to DataFrame
input_data = pd.DataFrame({
    "fim_motor": [fim_motor],
    "duration_since_stroke_onset": [duration_since_stroke_onset],
    "fim_cognition": [fim_cognition],
    "japan_coma_scale": [japan_coma_scale]
})

# Ensure column names match the scaler's expected input
expected_features = ["fim_motor", "fim_cognition", "duration_since_stroke_onset", "japan_coma_scale"]
if list(input_data.columns) != expected_features:
    st.error("Input features do not match the expected feature names!")
    st.write(f"Expected: {expected_features}")
    st.write(f"Received: {list(input_data.columns)}")
else:
    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Execute prediction
    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        st.subheader("Prediction Result")
        st.write(f"Predicted FILS score: {prediction[0]:.2f}")

