import streamlit as st
import joblib
import pandas as pd

# Load the saved model and related objects
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app configuration
st.title("Dysphagia Screening Application")
st.write("Enter the following features to predict the value of the FILS score.")

# User input fields with integer-only input
fim_motor = st.number_input("FIM Motor", min_value=13, max_value=91, value=13, step=1)
duration_since_stroke_onset = st.number_input("Duration Since Stroke Onset (days)", min_value=0, max_value=365, value=0, step=1)
fim_cognition = st.number_input("FIM Cognition", min_value=5, max_value=35, value=5, step=1)
japan_coma_scale = st.number_input("Japan Coma Scale", min_value=0, max_value=300, value=0, step=1)

# Convert input values to DataFrame
input_data = pd.DataFrame({
    "fim_motor": [fim_motor],
    "duration_since_stroke_onset": [duration_since_stroke_onset],
    "fim_cognition": [fim_cognition],
    "japan_coma_scale": [japan_coma_scale]
})

# Ensure column names match the scaler's expected input
expected_features = ["fim_motor", "fim_cognition", "duration_since_stroke_onset", "japan_coma_scale"]

# Adjust column order to match expected features
try:
    input_data = input_data[expected_features]
except KeyError:
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
        st.write(f"Predicted FILS score: {round(prediction[0], 1)}")
