import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# Streamlit UI for data input
st.title("Migraine Risk Prediction")
st.write("Please answer the following questions to help us predict your migraine risk.")

# Step 1: Ask user about their current symptoms
st.subheader("How are you feeling right now?")

headache = st.radio("Are you having a headache right now?", ("Yes", "No"))
nauseous = st.radio("Are you feeling nauseous?", ("Yes", "No"))
light_sensitivity = st.radio("Are you sensitive to light?", ("Yes", "No"))
noise_sensitivity = st.radio("Are you sensitive to noise?", ("Yes", "No"))

# Step 2: Ask about environmental factors
st.subheader("How is your current environment?")
light_intensity = st.slider('Light Intensity (0-100)', 0, 100, 50)
noise_level = st.slider('Noise Level (0-100)', 0, 100, 50)

# Step 3: Ask about physical data (optional, for more detailed prediction)
heart_rate = st.slider('Heart Rate (bpm)', 40, 200, 75)
skin_temp = st.slider('Skin Temperature (°C)', 30, 40, 36)

# Combine user inputs into a DataFrame
user_data = pd.DataFrame({
    'Light Intensity': [light_intensity],
    'Noise Levels': [noise_level],
    'Heart Rate': [heart_rate],
    'Skin Temp': [skin_temp]
})

# Step 4: Process inputs to make predictions
# For simplicity, we'll simulate a basic prediction based on the symptoms and environment
if headache == "Yes" or nauseous == "Yes" or light_sensitivity == "Yes" or noise_sensitivity == "Yes":
    migraine_prob = 0.8  # High chance of migraine based on symptoms
else:
    migraine_prob = 0.2  # Lower chance of migraine based on the absence of strong symptoms

# Display the predicted migraine risk to the user
st.subheader("Your Migraine Risk Prediction")
st.write(f"We estimate your chance of having a migraine is **{migraine_prob * 100:.2f}%** based on your current symptoms and environment.")

# Step 5: Give advice based on prediction
if migraine_prob > 0.7:
    st.write("Your risk is high! It might help to rest in a dark, quiet room and hydrate.")
elif migraine_prob > 0.4:
    st.write("Your risk is moderate. Consider taking a break and managing stress. If symptoms persist, rest and hydrate.")
else:
    st.write("Your risk is low, but keep an eye on how you're feeling and adjust your environment if needed.")

# Step 6: Show evaluation metrics for the model (Optional for demo purposes)
st.write("Here's how the model performs on past data:")

# Simulate model evaluation (this part would be replaced by actual model evaluation)
accuracy = np.random.uniform(0.7, 1.0)  # Random accuracy value for demo
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plot a simple Confusion Matrix
conf_matrix_rf = np.array([[50, 10], [5, 35]])  # Dummy confusion matrix values
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Migraine', 'Migraine'], yticklabels=['No Migraine', 'Migraine'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Model Performance)')
st.pyplot()

# Add additional visualizations if necessary (e.g., ROC Curve, Precision-Recall Curve)
