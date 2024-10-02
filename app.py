import streamlit as st
import numpy as np
import joblib
from keras.models import load_model

# Load the model and scaler
model = load_model('lstm_machine_failure_model.keras')
scaler = joblib.load('scaler.pkl')

# Title of the app
st.title('Machine Failure Prediction')

# Input features
setting1 = st.number_input('Coefficient of Setting 1')
setting2 = st.number_input('Coefficient of Setting 2')
setting3 = st.number_input('Coefficient of Setting 3')
s1 = st.number_input('Sensor 1')
s2 = st.number_input('Sensor 2')
s3 = st.number_input('Sensor 3')
s4 = st.number_input('Sensor 4')
s5 = st.number_input('Sensor 5')
s6 = st.number_input('Sensor 6')

# Button to predict
if st.button('Predict'):
    # Create input array
    input_features = np.array([[setting1, setting2, setting3, s1, s2, s3, s4, s5, s6]])

    # Scale the input features
    input_features_scaled = scaler.transform(input_features)

    # Reshape for LSTM
    # Ensure the sequence length matches your model's expectations
    input_features_scaled = input_features_scaled.reshape((1, 1, 9))  # (batch_size, seq_length, num_features)

    # Make prediction
    prediction = model.predict(input_features_scaled)
    predicted_label = (prediction > 0.5).astype("int32")

    # Display result
    if predicted_label[0][0] == 1:
        # Displaying the error message
        st.markdown(
            f"<span style='font-size: 20px; font-weight: bold; color: red;'>Machine will fail soon.</span>",
            unsafe_allow_html=True
        )
        
        # Displaying the probability message
        st.markdown(
            f"<span style='font-size: 20px; font-weight: bold; color: red;'>Probability within 30 days is {round(prediction[0][0] * 100, 2)}%</span>",
            unsafe_allow_html=True
        )
    else:
        # Displaying the success message for machine condition
        st.markdown(
            "<span style='font-size: 20px; font-weight: bold; color: green;'>Machine is in good condition.</span>",
            unsafe_allow_html=True
        )
        
        # Displaying the probability message
        st.markdown(
            f"<span style='font-size: 20px; font-weight: bold; color: green;'>Probability within 30 days is {round(prediction[0][0] * 100, 2)}%</span>",
            unsafe_allow_html=True
        )
