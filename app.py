import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('./logistic_regression_best_model.pkl')
scaler = joblib.load('./scaler.pkl')

# Streamlit app
st.title('Predicting the necessity for tube feeding in acute stroke patients')

# Input form
with st.form(key='input_form'):
    fim_m = st.number_input('Enter Motor Functional Independence Measure value', format='%f')
    fim_c = st.number_input('Enter Cognitive Functional Independence Measure value', format='%f')
    si = st.number_input('Enter Speech Intelligibility value',format='%f')
    submit_button = st.form_submit_button(label='Predict')

# Prediction
if submit_button:
    input_data = np.array([[fim_m, fim_c, si]])
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    # Display the result
    if prediction[0] == 0:
        st.write('Tube feeding is not necessary')
    else:
        st.write('Tube feeding is necessary')
