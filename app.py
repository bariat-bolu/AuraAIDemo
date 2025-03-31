import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic data
def generate_data(num_samples=1000):
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=num_samples, freq='H')
    heart_rate = np.random.randint(60, 100, num_samples)
    skin_temp = np.random.uniform(36, 37, num_samples)
    eda = np.random.uniform(0, 100, num_samples)
    migraine_frequency = np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
    migraine_intensity = np.random.randint(1, 11, num_samples)
    
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Heart Rate': heart_rate,
        'Skin Temp': skin_temp,
        'EDA': eda,
        'Migraine Frequency': migraine_frequency,
        'Migraine Intensity': migraine_intensity,
    })
    df['Migraine Onset'] = np.where(
        (df['Migraine Frequency'] == 1) & (df['Migraine Intensity'] > 5), 1, 0
    )
    return df

df = generate_data()
features = ['Heart Rate', 'Skin Temp', 'EDA']
X = df[features]
y = df['Migraine Onset']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("Migraine Prediction App")
st.write("Enter your physiological data to predict migraine onset.")

heart_rate = st.slider("Heart Rate", 50, 120, 75)
skin_temp = st.slider("Skin Temperature (°C)", 35.0, 38.0, 36.5)
eda = st.slider("Electrodermal Activity (EDA)", 0, 100, 50)

if st.button("Predict Migraine Onset"):
    input_data = pd.DataFrame([[heart_rate, skin_temp, eda]], columns=features)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.write(f"Predicted Migraine Onset: {'Yes' if prediction[0] == 1 else 'No'}")

# Show data visualization
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df[features + ['Migraine Onset']].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
