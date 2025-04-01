import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# File to store logged entries
DATA_FILE = "migraine_logs.csv"

# Load existing data if available
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["Date", "Time", "Symptoms", "Trigger", "Medication"])

# Store the DataFrame in session state
if "log_data" not in st.session_state:
    st.session_state.log_data = df

# App title
st.title("Migraine Prediction App")

# Sidebar Navigation
menu = ["Home", "Log Entry", "Predictions/Insights", "Reports/History", "Settings/Profile"]
choice = st.sidebar.radio("Select a page", menu)

# Home/Dashboard Screen
if choice == "Home":
    st.header("Current Prediction Status")

    # Display recent entries from session state
    st.subheader("Recent Entries")
    if not st.session_state.log_data.empty:
        st.table(st.session_state.log_data.tail(5))  # Show last 5 entries
    else:
        st.write("No records found. Log your symptoms first!")

# Log Entry Screen
elif choice == "Log Entry":
    st.header("Log Your Symptoms")

    date_input = st.date_input("Date", datetime.today())
    time_input = st.time_input("Time", datetime.now().time())
    symptom = st.text_area("Describe your symptoms:")
    trigger = st.text_area("What triggered your migraine?")
    medication = st.text_area("Medication taken:")

    if st.button("Log Entry"):
        new_entry = pd.DataFrame({
            "Date": [date_input],
            "Time": [time_input.strftime("%H:%M")],
            "Symptoms": [symptom],
            "Trigger": [trigger],
            "Medication": [medication]
        })

        # Update session state
        st.session_state.log_data = pd.concat([st.session_state.log_data, new_entry], ignore_index=True)

        # Save to CSV
        st.session_state.log_data.to_csv(DATA_FILE, index=False)

        st.success("Entry Logged Successfully!")

# Reports/History Screen
elif choice == "Reports/History":
    st.header("Your Migraine History")
    if not st.session_state.log_data.empty:
        st.table(st.session_state.log_data)  # Show full history
    else:
        st.write("No records found.")

# Settings/Profile Screen
elif choice == "Settings/Profile":
    st.header("User Profile and Settings")
    username = st.text_input("Your Name:")
    email = st.text_input("Email:")
    notifications = st.checkbox("Enable notifications?")

    if st.button("Save Settings"):
        st.write("Settings saved!")
