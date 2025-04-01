import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('migraine_model.pkl')

# Function to make predictions
def predict_migraine(nausea, light, noise):
    features = np.array([[nausea, light, noise]])
    prediction = model.predict(features)
    return prediction[0]

# Function to display the recommendation based on prediction
def get_recommendation(prediction, nausea, light, noise):
    recommendations = ""
    if prediction == "High":
        recommendations += "Your migraine risk is high. It might help to: \n"
        recommendations += "- Rest in a dark, quiet room\n"
        recommendations += "- Hydrate with water\n"
        recommendations += "- Take medication if you have one prescribed\n"
        if nausea >= 7:
            recommendations += "- Try to eat something light to ease the nausea\n"
        if light >= 7:
            recommendations += "- Avoid screens and bright lights\n"
        if noise >= 7:
            recommendations += "- Wear noise-canceling headphones or earplugs\n"
    elif prediction == "Moderate":
        recommendations += "Your migraine risk is moderate. You may want to: \n"
        recommendations += "- Take a short rest in a calm environment\n"
        recommendations += "- Stay hydrated\n"
        if nausea >= 6:
            recommendations += "- Try drinking ginger tea to ease nausea\n"
        if light >= 6:
            recommendations += "- Dim the lights or wear sunglasses\n"
        if noise >= 6:
            recommendations += "- Reduce background noise\n"
    else:
        recommendations += "Your migraine risk is low. Continue monitoring, but you're likely safe today.\n"
        recommendations += "- Drink plenty of water\n"
        if nausea >= 5:
            recommendations += "- Consider taking anti-nausea medication if needed\n"
        if light >= 5:
            recommendations += "- Keep lighting moderate\n"
        if noise >= 5:
            recommendations += "- Avoid noisy environments if possible\n"
    
    return recommendations

# Streamlit app
st.title("Migraine Prediction and Management")

# Collect user inputs
st.header("Rate Your Symptoms (Scale 0-10)")

nausea = st.slider("Are you feeling nauseous?", 0, 10, 0)
light = st.slider("Is light bothering you?", 0, 10, 0)
noise = st.slider("Is noise bothering you? Is everything excessively loud?", 0, 10, 0)

# Make prediction based on inputs
prediction = predict_migraine(nausea, light, noise)

# Display the prediction
st.subheader("Prediction:")
st.markdown(f"### **{prediction} Risk**", unsafe_allow_html=True)

# Display recommendations based on prediction
recommendations = get_recommendation(prediction, nausea, light, noise)
st.subheader("Recommendations:")
st.write(recommendations)
