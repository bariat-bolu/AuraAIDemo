import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Mock prediction function (Replace with your actual prediction logic)
def predict_migraine(severity, nausea, light_sensitivity, sound_sensitivity, is_migraine):
    if is_migraine:
        if severity >= 7 and (nausea >= 5 or light_sensitivity >= 5 or sound_sensitivity >= 5):
            return {"risk_level": "High", "recommendation": "Rest in a dark, quiet room and hydrate."}
        elif severity >= 5:
            return {"risk_level": "Moderate", "recommendation": "Consider taking medication and resting."}
        else:
            return {"risk_level": "Low", "recommendation": "Keep track of your symptoms and stay hydrated."}
    else:
        if severity >= 6:
            return {"risk_level": "Moderate", "recommendation": "Consider resting and hydration."}
        else:
            return {"risk_level": "Low", "recommendation": "Take a break and stay hydrated."}

# App title
st.title("Migraine Prediction App")

# Description of the app
st.write("Track your migraine symptoms and get predictions based on your data.")

# Core Navigation
menu = ["Home", "Log Entry", "Predictions/Insights", "Reports/History", "Settings/Profile"]
choice = st.sidebar.selectbox("Select a page", menu)

# Home/Dashboard Screen
if choice == "Home":
    st.header("Current Prediction Status")
    
    # Example input fields for symptoms
    headache_severity = st.slider("Headache Severity (1-10)", 0, 10, 5)
    
    st.subheader("Rate Your Symptoms")
    
    # Rate nausea, light sensitivity, and sound sensitivity
    nausea = st.slider("Are you feeling nauseous? Rate it on a scale of 0 to 10.", 0, 10, 5)
    light_sensitivity = st.slider("Is light bothering you? Rate it on a scale of 0 to 10.", 0, 10, 5)
    sound_sensitivity = st.slider("Is noise bothering you? Is everything excessively loud? Rate it on a scale of 0 to 10.", 0, 10, 5)
    
    # Add a section where users can identify if it's a migraine or a bad headache
    is_migraine = st.radio("Are you experiencing a migraine or just a bad headache?", 
                           ("Migraine", "Bad Headache"))
    
    # Prediction based on symptoms and whether it's a migraine or headache
    prediction = predict_migraine(headache_severity, nausea, light_sensitivity, sound_sensitivity, is_migraine == "Migraine")
    
    # Display prediction result with larger text for risk level
    st.markdown(f"### **Prediction: {prediction['risk_level']} Risk**", unsafe_allow_html=True)
    st.write(f"Recommendation: {prediction['recommendation']}")
    
    # Recent entries (for example purposes)
    st.subheader("Recent Entries")
    # Example: Displaying a table of recent log entries (use your database or pandas DataFrame)
    recent_entries = pd.DataFrame({
        "Date": [datetime.now() - pd.Timedelta(days=i) for i in range(3)],
        "Symptoms": ["Mild headache", "Moderate headache", "Severe headache"],
        "Risk Level": ["Low", "Moderate", "High"]
    })
    st.table(recent_entries)

# Other screens (Log Entry, Predictions/Insights, etc.) would be similar to what we previously discussed.
