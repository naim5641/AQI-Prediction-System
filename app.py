import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Advanced AQI Analytics", page_icon="📊", layout="wide")

# --- LOAD MODEL & DATA (Placeholder for EDA) ---
@st.cache_resource
def load_assets():
    model = joblib.load('aqi_xgboost_model.pkl')
    # EDA এর জন্য একটি স্যাম্পল ডাটাসেট লোড করা হচ্ছে (আপনার অরিজিনাল ফাইলটি এখানে দিন)
    df = pd.read_csv('data_preprocecing_24.csv') 
    # আপাতত ডেমো ডাটা তৈরি করছি গ্রাফ দেখানোর জন্য
    df = pd.DataFrame(np.random.randint(0,200,size=(100, 6)), columns=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'])
    df['AQI'] = df['PM2.5'] * 1.5 + np.random.randint(0,20,100)
    df['Hour'] = np.random.randint(0,24,100)
    return model, df

model, df = load_assets()

# --- HELPER FUNCTIONS ---
def get_live_data(city):
    API_TOKEN = "8336110b0e0345c68d0dc9ca4554b7122e5ad33d"
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    try:
        r = requests.get(url).json()
        if r['status'] == 'ok':
            iaqi = r['data']['iaqi']
            return {g: iaqi.get(g, {}).get('v', 0) for g in ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']}, True
        return None, False
    except: return None, False

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("💎 AQI Expert System")
page = st.sidebar.radio("Menu", ["Live Prediction", "What-If Simulator", "Data Insights (EDA)", "Model Performance"])

# ---------------- PAGE 1: LIVE PREDICTION ----------------
if page == "Live Prediction":
    st.title("🏙️ Real-Time AQI Forecast & Alerts")
    selected_city = st.selectbox("Select City", ["Dhaka", "Chittagong", "Cumilla", "Sylhet", "Rajshahi"])
    
    if st.button("Fetch & Predict"):
        data, success = get_live_data(selected_city)
        if success:
            now = datetime.now()
            features = np.array([[data['so2'], data['no2'], data['co'], data['o3'], data['pm25'], data['pm10'], 
                                 now.year, now.month, now.day, now.hour, now.weekday(), 
                                 (1 if now.weekday() >= 5 else 0), data['pm25']*1.1, data['pm25']*1.0]])
            
            pred = model.predict(features)[0]
            
            # Health Alert logic
            col1, col2 = st.columns(2)
            col1.metric("Predicted AQI", f"{pred:.2f}")
            
            if pred > 150:
                st.error("🚨 DANGER: আজ শিশুদের এবং শ্বাসকষ্টজনিত রোগীদের বাইরে নিয়ে যাওয়া থেকে বিরত থাকুন।")
            
            # Comparison with WHO
            who_limit = 15 # PM2.5 daily limit
            times_higher = data['pm25'] / who_limit
            st.warning(f"⚠️ বর্তমান PM2.5 মান WHO স্ট্যান্ডার্ডের চেয়ে {times_higher:.1f} গুণ বেশি!")

# ---------------- PAGE 2: WHAT-IF SIMULATOR ----------------
elif page == "What-If Simulator":
    st.title("🧪 Scenario Simulator")
    st.write("গ্যাসের মান পরিবর্তন করে দেখুন AQI-তে কী প্রভাব পড়ে।")
    
    s_pm25 = st.slider("PM2.5 Level", 0, 500, 50)
    s_pm10 = st.slider("PM10 Level", 0, 500, 100)
    
    test_features = np.array([[10, 30, 1.0, 20, s_pm25, s_pm10, 2026, 4, 23, 12, 3, 0, 100, 100]])
    sim_pred = model.predict(test_features)[0]
    
    st.subheader(f"Predicted AQI: {sim_pred:.2f}")
    st.info("💡 টিপস: দেখুন PM2.5 কমালে AQI কত দ্রুত নিচে নেমে আসে!")

# ---------------- PAGE 3: DATA INSIGHTS (EDA) ----------------
elif page == "Data Insights (EDA)":
    st.title("📊 Training Data Analytics")
    
    tab1, tab2 = st.tabs(["Correlation", "Trends"])
    
    with tab1:
        st.subheader("Correlation Heatmap")
        fig_corr = px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr)
        
    with tab2:
        st.subheader("Peak Hour Analysis")
        hourly_aqi = df.groupby('Hour')['AQI'].mean().reset_index()
        fig_hour = px.line(hourly_aqi, x='Hour', y='AQI', title="দিনের কোন সময়ে দূষণ বেশি?")
        st.plotly_chart(fig_hour)

# ---------------- PAGE 4: MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.title("🧠 Model Evaluation Metrics")
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Model Accuracy (R2 Score)", "0.94") # আপনার আসল স্কোর দিন
    m_col2.metric("Mean Absolute Error (MAE)", "5.21")
    
    st.subheader("Feature Importance")
    # এটি আপনার মডেলের ওপর ভিত্তি করে অটোমেটিক হবে
    importance = pd.DataFrame({'Feature': ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'Hour'], 
                               'Value': model.feature_importances_[:6]})
    fig_imp = px.bar(importance.sort_values('Value'), x='Value', y='Feature', orientation='h')
    st.plotly_chart(fig_imp)

st.divider()
st.caption("Developed by Naim | CSE Student | End-to-End AQI Project")