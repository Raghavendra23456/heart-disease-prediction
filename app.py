import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load pre-trained models (assuming models are saved in .h5 format)
ecg_model = load_model('ecg_model.h5')
present_model = load_model('random_forest_model.h5')

# Function to make predictions using the ECG model
def predict_ecg(data):
    data_scaled = scaler_ecg.transform(data)  # Assuming the ECG data needs scaling
    prediction = ecg_model.predict(data_scaled)
    confidence = prediction[0][0]  # Probability of class 1 (abnormal)
    return confidence

# Function to make predictions using the present model (14-column dataset)
def predict_present_model(data):
    data_scaled = scaler_present.transform(data)  # Assuming scaling is required
    prediction = present_model.predict(data_scaled)
    confidence = prediction[0][0]  # Probability of class 1 (abnormal)
    return confidence

# Title of the Streamlit App
st.title("ECG & Health Prediction App")

# Section for ECG Model Predictions
st.header("ECG Data Prediction")
uploaded_ecg_file = st.file_uploader("Upload ECG Data (CSV file)", type="csv")
if uploaded_ecg_file is not None:
    ecg_data = pd.read_csv(uploaded_ecg_file, header=None)
    st.write("Sample ECG Data:", ecg_data.head())
    
    if st.button("Predict ECG"):
        ecg_confidence = predict_ecg(ecg_data.values)
        st.write(f"ECG Model Prediction Confidence for Class 1 (Abnormal): {ecg_confidence:.4f}")
        st.write(f"Confidence for Class 0 (Normal): {1 - ecg_confidence:.4f}")

# Section for Present Model Predictions
st.header("Health Data Prediction")
uploaded_health_file = st.file_uploader("Upload Health Data (CSV file)", type="csv")
if uploaded_health_file is not None:
    health_data = pd.read_csv(uploaded_health_file)
    st.write("Sample Health Data:", health_data.head())

    if st.button("Predict Health"):
        health_confidence = predict_present_model(health_data.values)
        st.write(f"Health Model Prediction Confidence for Class 1 (Abnormal): {health_confidence:.4f}")
        st.write(f"Confidence for Class 0 (Normal): {1 - health_confidence:.4f}")

# Add instructions or notes
st.write("Upload appropriate CSV files to get predictions. The ECG model expects data similar to the MIT-BIH dataset.")
