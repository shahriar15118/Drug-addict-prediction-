import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('best_model.pkl')  # Make sure this file is in the same directory

# Set up the Streamlit interface
st.title("Drug Addiction Prediction")

# Collect user inputs
age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", options=[0, 1], index=1)  # 0 for Female, 1 for Male
occupation = st.number_input("Occupation", min_value=0, max_value=10, value=3)

# Add other input fields based on your dataset (same as before)

# Collect input features
input_features = [age, gender, occupation]  # Include other features

# Predict button
if st.button("Predict"):
    prediction = model.predict([input_features])  # Wrap input as a 2D array
    probabilities = model.predict_proba([input_features])  # For probabilities
    st.write(f"Prediction: {'Drug Addict' if prediction[0] == 1 else 'Not Drug Addict'}")
    st.write(f"Probabilities: {probabilities[0]}")
