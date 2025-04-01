import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Assuming your trained model is in a function like 'predict_migraine'
# from your_model import predict_migraine  # This is where you import your trained model function

# App title
st.title("Migraine Prediction App")

# Add a description of the app
st.write("Track your migraine symptoms and get predictions based on your data.")

# Core Navigation
menu = ["Home", "Log Entry", "Predictions/Insights", "Reports/History", "Settings/Profile"]
choice = st.sidebar.selectbox("Select a page", menu)

# Mock prediction function to demonstrate
# Replace this with your actual model's prediction function
def predict_migraine(severity, nausea, light_sensitivity, sound_sensitivity):
    # Replace this with your actual model prediction logic
    if severity >= 7 and (nausea >= 5 or light_sensitivity >= 5 or sound_sensitivity >= 5):
        return {"risk_level": "High", "recommendation": "Rest in a dark, quiet room and hydrate."}
    elif severity >= 5:
        return {"risk_level": "Moderate", "recommendation": "Consider taking medication and resting."}
    else:
        return {"risk_level": "Low", "recommendation": "Keep track of your symptoms and stay hydrated."}

# Home/Dashboard Screen
if choice == "Home":
    st.header("Current Prediction Status")
    
    # Example input fields for symptoms, etc.
    headache_severity = st.slider("Headache Severity (1-10)", 1, 10, 5)
    
    st.subheader("Rate Your Symptoms")
    
    # Rate nausea, light sensitivity, and sound sensitivity on a 0-10 scale
    nausea = st.slider("Are you feeling nauseous?", 0, 10, 5)
    light_sensitivity = st.slider("Is light bothering you?", 0, 10, 5)
    sound_sensitivity = st.slider("Is noise bothering you? Is everything excessively loud?", 0, 10, 5)
    
    # Prediction based on symptoms
    prediction = predict_migraine(headache_severity, nausea, light_sensitivity, sound_sensitivity)
    
    # Display prediction result
    st.markdown(f"### **Prediction: {prediction['risk_level']} Risk**", unsafe_allow_html=True)
    st.write(f"Recommendation: {prediction['recommendation']}")
    
    # Recent entries
    st.subheader("Recent Entries")
    # Example: Displaying a table of recent log entries (use your database or pandas DataFrame)
    recent_entries = pd.DataFrame({
        "Date": [datetime.now() - pd.Timedelta(days=i) for i in range(3)],
        "Symptoms": ["Mild headache", "Moderate headache", "Severe headache"],
        "Risk Level": ["Low", "Moderate", "High"]
    })
    st.table(recent_entries)

# Log/Entry Screen
elif choice == "Log Entry":
    st.header("Log Your Symptoms")
    
    date_input = st.date_input("Date", datetime.today())
    time_input = st.time_input("Time", datetime.now().time())
    
    symptom = st.text_input("Describe your symptoms:")
    trigger = st.text_input("What triggered your migraine?")
    medication = st.text_input("Medication taken:")
    
    # Option to log symptoms
    if st.button("Log Entry"):
        st.write("Entry Logged!")
        # Add functionality to store this in your database or file

# Predictions/Insights Screen
elif choice == "Predictions/Insights":
    st.header("Predicted Migraine Risk")
    
    # Example: Prediction history with a graph (use your data)
    risk_data = pd.DataFrame({
        "Date": [datetime.now() - pd.Timedelta(days=i) for i in range(10)],
        "Risk Level": np.random.choice(['Low', 'Moderate', 'High'], size=10)
    })
    st.line_chart(risk_data.set_index('Date')['Risk Level'].apply(lambda x: {'Low': 0, 'Moderate': 1, 'High': 2}[x]))

# Reports/History Screen
elif choice == "Reports/History":
    st.header("Historical Reports")
    
    # Display a sample report
    report_data = pd.DataFrame({
        "Date": [datetime.now() - pd.Timedelta(days=i) for i in range(10)],
        "Frequency": np.random.randint(1, 10, 10),
        "Intensity": np.random.randint(1, 10, 10)
    })
    
    st.write("Your Migraine History Report:")
    st.table(report_data)
    
    # Option to export report
    if st.button("Download Report"):
        # Add functionality to download the report
        st.write("Report ready to download!")

# Settings/Profile Screen
elif choice == "Settings/Profile":
    st.header("User Profile and Settings")
    
    username = st.text_input("Your Name:")
    email = st.text_input("Email:")
    
    # Custom notifications or other preferences
    notifications = st.checkbox("Enable notifications?")
    
    if st.button("Save Settings"):
        st.write("Settings saved!")
