
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🏙️",
    layout="centered"
)

# 2. Load the trained Random Forest model
@st.cache_resource
def load_model():
    try:
        return joblib.load('aqi_xgboost_model.pkl')
    except:
        st.error("Model file not found! Please ensure 'aqi_xgboost_model.pkl' is in the repository.")
        return None

model = load_model()

# 3. Application UI Header
st.title("🏙️ Air Quality Index (AQI) Prediction")
st.markdown("""
This application predicts the **AQI** based on gas concentration levels and temporal features using a **Random Forest Regressor** model.
""")

st.info("Please enter the following parameters to get a real-time AQI prediction.")

# 4. Input Fields (Organized in Two Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Gas Concentrations")
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=50.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=100.0)
    no2 = st.number_input("NO2 (ppb)", min_value=0.0, value=30.0)
    so2 = st.number_input("SO2 (ppb)", min_value=0.0, value=10.0)
    co = st.number_input("CO (mg/m³)", min_value=0.0, value=1.0)
    o3 = st.number_input("O3 (ppb)", min_value=0.0, value=20.0)

with col2:
    st.subheader("📅 Time & Context")
    year = st.selectbox("Year", [2025, 2026])
    month = st.slider("Month", 1, 12, 4)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour (24h format)", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0,1,2,3,4,5,6])
    is_weekend = st.selectbox("Is it Weekend? (0=No, 1=Yes)", [0, 1])

# Sidebar for Historical/Lag Features
st.sidebar.header("🔄 Historical Data")
aqi_moving_avg = st.sidebar.number_input("Moving Average AQI (Last 24h)", value=100.0)
aqi_lag_1 = st.sidebar.number_input("Previous Hour AQI (Lag 1)", value=100.0)

# 5. Prediction Logic
if st.button("🚀 Predict AQI Now"):
    if model is not None:
        # Input features in correct order
        input_data = np.array([[so2, no2, co, o3, pm25, pm10, year, month, day, hour, day_of_week, is_weekend, aqi_moving_avg, aqi_lag_1]])

        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader(f"📊 Predicted AQI Value: {prediction:.2f}")

        if prediction <= 50:
            st.success("🟢 Air Quality: **GOOD**")
        elif prediction <= 100:
            st.warning("🟡 Air Quality: **MODERATE**")
        elif prediction <= 150:
            st.error("🟠 Air Quality: **UNHEALTHY**")
        else:
            st.error("🔴 Air Quality: **POOR/HAZARDOUS**")

        st.balloons()
    else:
        st.error("Model is not loaded properly.")

st.markdown("---")
st.caption("Developed as a Machine Learning Portfolio Project.")
