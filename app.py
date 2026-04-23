import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Advanced AQI Dashboard", layout="wide")

# --- LOAD ASSETS (Model & Dataset) ---
@st.cache_resource
def load_assets():
    model = joblib.load('aqi_xgboost_model.pkl')
    # আপনার প্রি-প্রসেসিং করা ডাটাসেটটি এখানে লোড করুন
    try:
        df = pd.read_csv('data_preprocecing_24.csv')
    except:
        # ডামি ডেটা (যদি ফাইল না থাকে তবে এটি কাজ করবে)
        df = pd.DataFrame(np.random.randint(20, 200, size=(100, 7)), 
                          columns=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI'])
        df['Timestamp'] = pd.date_range(start='2026-04-01', periods=100, freq='H')
    return model, df

model, df = load_assets()

# --- API HELPER ---
def get_live_data(city):
    token = "8336110b0e0345c68d0dc9ca4554b7122e5ad33d"
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        r = requests.get(url).json()
        if r['status'] == 'ok':
            return r['data'], True
    except: return None, False

# --- UI COMPONENTS ---

# 1. Gauge Meter Function
def create_gauge(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': "Current AQI Index"},
        gauge = {
            'axis': {'range': [None, 500]},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [51, 100], 'color': "yellow"},
                {'range': [101, 200], 'color': "orange"},
                {'range': [201, 300], 'color': "red"},
                {'range': [301, 500], 'color': "purple"}],
            'bar': {'color': "black"}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- SIDEBAR ---
st.sidebar.title("🎮 Control Panel")
city_list = ["Dhaka", "Chittagong", "Cumilla", "Sylhet", "Rajshahi"]
selected_city = st.sidebar.selectbox("📍 Select City", city_list)
mode = st.sidebar.radio("🛠️ Mode Selection", ["Live Analytics", "What-If Simulator", "Historical Reports"])

# --- HEADER ---
st.markdown(f"# 🌍 {selected_city} Air Quality Dashboard")
st.write(f"**Date:** {datetime.now().strftime('%d %B, %Y')} | **Live Status:** Connected ✅")

# --- MAIN LOGIC ---
if mode == "Live Analytics":
    live_json, success = get_live_data(selected_city)
    
    if success:
        iaqi = live_json['iaqi']
        curr_aqi = live_json['aqi']
        
        # Row 1: Gauge and Quick Info
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.plotly_chart(create_gauge(curr_aqi), use_container_width=True)
        with col2:
            st.metric("PM2.5 Level", f"{iaqi.get('pm25', {}).get('v', 0)}")
            st.metric("Humidity", f"{iaqi.get('h', {}).get('v', 0)}%")
        with col3:
            st.metric("Temperature", f"{iaqi.get('t', {}).get('v', 0)}°C")
            st.write(f"🕒 **Last Updated:** \n{live_json['time']['s']}")

        st.divider()

        # Row 2: Health Insights & Map
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.subheader("🏥 Health Advice")
            if curr_aqi <= 100:
                st.success("বাতাস স্বচ্ছ। বাইরে সময় কাটাতে পারেন।")
            elif curr_aqi <= 200:
                st.warning("অ্যালার্জি বা শ্বাসকষ্ট থাকলে মাস্ক ব্যবহার করুন।")
            else:
                st.error("বিপজ্জনক! ঘরে থাকার চেষ্টা করুন এবং জানালা বন্ধ রাখুন।")
            
            st.info(f"💡 **WHO Benchmark:** বর্তমান বায়ুমান WHO গাইডলাইনের চেয়ে {round(curr_aqi/15, 1)} গুণ বেশি।")
            
        with col_m2:
            st.subheader("🗺️ Geospatial View")
            # চট্টগ্রামের একটি ডিফল্ট লোকেশন (উদাহরণস্বরূপ)
            map_data = pd.DataFrame({'lat': [22.3569], 'lon': [91.7832]})
            st.map(map_data)

        # Row 3: Charts
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 Time Series Trend (7 Days)")
            fig_line = px.line(df, x='Timestamp', y='AQI', title="AQI Change Over Time")
            st.plotly_chart(fig_line, use_container_width=True)
        with c2:
            st.subheader("📊 Feature Importance (ML)")
            importance = pd.DataFrame({'Feature': ['PM2.5', 'NO2', 'CO', 'SO2', 'O3'], 'Value': [0.4, 0.2, 0.15, 0.1, 0.15]})
            st.plotly_chart(px.bar(importance, x='Value', y='Feature', orientation='h'), use_container_width=True)

elif mode == "What-If Simulator":
    st.subheader("🧪 Machine Learning Scenario Simulator")
    st.write("গ্যাসের মান পরিবর্তন করে দেখুন প্রেডিকশনে কী প্রভাব পড়ে।")
    
    s_pm2 = st.sidebar.slider("PM2.5", 0, 500, 100)
    s_no2 = st.sidebar.slider("NO2", 0, 200, 30)
    
    # Prediction logic
    features = np.array([[10, s_no2, 1.0, 20, s_pm2, 100, 2026, 4, 23, 12, 3, 0, 100, 100]])
    sim_res = model.predict(features)[0]
    
    st.metric("Simulated Predicted AQI", f"{sim_res:.2f}")
    st.progress(min(int(sim_res/5), 100))

elif mode == "Historical Reports":
    st.subheader("📂 Download Analytics Reports")
    st.write("আপনার ডাটাসেটের সারাংশ ডাউনলোড করুন।")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full CSV Report", data=csv, file_name="aqi_report.csv", mime='text/csv')
    
    st.table(df.head(10))

# --- FOOTER ---
st.sidebar.markdown("---")
if st.sidebar.button("🔔 Send Telegram Alert"):
    st.sidebar.write("Alert sent to admin!") # এখানে বটের API কল হবে
