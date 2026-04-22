
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Page Configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🏙️",
    layout="centered"
)

# 2. Load the trained XGBoost model
@st.cache_resource
def load_model():
    try:
        # Loading the specific XGBoost model file
        return joblib.load('aqi_xgboost_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Ensure 'aqi_xgboost_model.pkl' is uploaded to the GitHub repository.")
        return None

model = load_model()

# 3. UI Header
st.title("🏙️ AQI Prediction (XGBoost)")
st.markdown("Predict Air Quality Index based on real-time pollutant data.")

# 4. Input Fields
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
    pm10 = st.number_input("PM10", min_value=0.0, value=100.0)
    no2 = st.number_input("NO2", min_value=0.0, value=30.0)
    so2 = st.number_input("SO2", min_value=0.0, value=10.0)
    co = st.number_input("CO", min_value=0.0, value=1.0)
    o3 = st.number_input("O3", min_value=0.0, value=20.0)

with col2:
    year = st.selectbox("Year", [2025, 2026])
    month = st.slider("Month", 1, 12, 4)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", [0,1,2,3,4,5,6])
    is_weekend = st.selectbox("Is Weekend?", [0, 1])

st.sidebar.header("Historical Context")
aqi_moving_avg = st.sidebar.number_input("Moving Avg AQI", value=100.0)
aqi_lag_1 = st.sidebar.number_input("Lag 1 AQI", value=100.0)

# 5. Prediction Logic
if st.button("🚀 Predict AQI"):
    if model is not None:
        # Feature order must match training data
        features = np.array([[so2, no2, co, o3, pm25, pm10, year, month, day, hour, day_of_week, is_weekend, aqi_moving_avg, aqi_lag_1]])
        prediction = model.predict(features)[0]
        
        st.markdown("---")
        st.subheader(f"📊 Predicted AQI: {prediction:.2f}")

        if prediction <= 50:
            st.success("🟢 Air Quality: GOOD")
        elif prediction <= 100:
            st.warning("🟡 Air Quality: MODERATE")
        elif prediction <= 150:
            st.error("🟠 Air Quality: UNHEALTHY")
        else:
            st.error("🔴 Air Quality: HAZARDOUS")
    else:
        st.error("Model file missing!")
