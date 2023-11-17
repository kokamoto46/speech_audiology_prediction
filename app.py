#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

# Load the trained model
@st.cache
def load_model():
    return joblib.load("./xgboost_best_model.pkl")

model = load_model()

# Streamlit app title
st.title("Machine Learning Model for Predicting the Requirement of Enteral Feeding in Acute Stroke Patients")

# Taking user input for the three features
st.sidebar.header("Input Features")

def user_input_features():
    fim_m = st.sidebar.number_input("FIM-M", min_value=13.0, max_value=91.0, value=91.0)
    fim_c = st.sidebar.number_input("FIM-C", min_value=5.0, max_value=35.0, value=35.0)
    si = st.sidebar.number_input("SI", min_value=1.0, max_value=5.0, value=5.0)
    return [fim_m, fim_c, si]

input_features = user_input_features()

# Predict button
if st.button('Predict'):
    prediction = model.predict(np.array([input_features]))[0]
    explanation = "0: tube feeding is not required" if prediction == 0 else "1: tube feeding is required"
    st.write(f"Prediction: {prediction}")
    st.write(explanation)

# Optional: Display the input features
st.write("## Input Features")
st.write(input_features)

