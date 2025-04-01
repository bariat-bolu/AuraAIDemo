import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Streamlit UI for data input
st.title("Migraine Risk Prediction")
st.write("Please answer the following questions to help us predict your migraine risk.")

# Step 1: Ask user about their current symptoms
st.subheader("How are you feeling right now?")

headache = st.radio("Are you having a headache right now?", ("Yes", "No"))
nauseous = st.radio("Are you feeling nauseous?", ("Yes", "No"))
sensitive_to_light = st.radio("Are you sensitive to light?", ("Yes", "No"))
sensitive_to_noise = st.radio("Are you sensitive to noise?", ("Yes", "No"))

# Step 2: Ask about environmental factors
st.subheader("How is your current environment?")
light_intensity = st.slider('Light Intensity (0-100)', 0, 100, 50)
noise_level = st.slider('Noise Level (0-100)', 0, 100, 50)

# Step 3: Ask about physical data (optional, for more detailed prediction)
heart_rate = st.slider('Heart Rate (beats per minute)', 40, 200, 75)
skin_temp = st.slider('Skin Temperature (°C)', 30, 40, 36)

# Combine user inputs into a DataFrame (for prediction)
user_data = pd.DataFrame({
    'Light Intensity': [light_intensity],
    'Noise Levels': [noise_level],
    'Heart Rate': [heart_rate],
    'Skin Temp': [skin_temp]
})

# Step 4: Make a basic prediction based on symptoms and environment
# Simple risk estimation based on symptoms
if headache == "Yes" or nauseous == "Yes" or sensitive_to_light == "Yes" or sensitive_to_noise == "Yes":
    migraine_risk = 0.8  # High chance of migraine
else:
    migraine_risk = 0.2  # Low chance of migraine

# Step 5: Display prediction to the user
st.subheader("Your Migraine Risk")
if migraine_risk > 0.7:
    st.write("You are at high risk of having a migraine.")
    st.write("We suggest resting in a dark, quiet room, avoiding screens, and drinking plenty of water.")
elif migraine_risk > 0.4:
    st.write("You have a moderate chance of having a migraine.")
    st.write("Taking a break, reducing stress, and staying hydrated may help.")
else:
    st.write("Your risk of having a migraine is low.")
    st.write("However, keep an eye on how you're feeling and rest if needed.")

# No confusion matrix or technical details
# Optional: Show a simple visual with no technical jargon
fig, ax = plt.subplots()
ax.bar(["Low Risk", "Moderate Risk", "High Risk"], [0.2, 0.5, 0.8], color=['green', 'yellow', 'red'])
ax.set_title("Migraine Risk Levels")
ax.set_ylabel("Risk Level")

st.pyplot(fig)
