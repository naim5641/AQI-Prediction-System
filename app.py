import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AQI Dataset Dashboard", layout="wide")

# --- LOAD ASSETS (Model & Dataset) ---
@st.cache_resource
def load_assets():
    # মডেল লোড
    model = joblib.load('aqi_xgboost_model.pkl')
    
    # ডাটাসেট লোড (নিশ্চিত করুন ফাইলের নাম সঠিক আছে)
    try:
        df = pd.read_csv('your_preprocessed_dataset.csv')
        # Timestamp কলামটি datetime ফরম্যাটে কনভার্ট করা
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        else:
            df['Timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
    except:
        # ফাইল না থাকলে ডামি ডেটা তৈরি
        df = pd.DataFrame(np.random.randint(20, 200, size=(100, 7)), 
                          columns=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI'])
        df['Timestamp'] = pd.date_range(start='2026-04-01', periods=100, freq='H')
    
    return model, df

model, df = load_assets()

# --- GAUGE METER FUNCTION ---
def create_gauge(value, title_text="AQI Index"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title_text},
        gauge = {
            'axis': {'range': [None, 500]},
            'steps': [
                {'range': [0, 50], 'color': "#00e400"},    # Good
                {'range': [51, 100], 'color': "#ffff00"},   # Moderate
                {'range': [101, 200], 'color': "#ff7e00"},  # Unhealthy for Sensitive
                {'range': [201, 300], 'color': "#ff0000"},  # Unhealthy
                {'range': [301, 500], 'color': "#8f3f97"}], # Hazardous
            'bar': {'color': "black"}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- SIDEBAR ---
st.sidebar.title("📊 AQI Analytics Control")
mode = st.sidebar.radio("🛠️ Select Feature", 
                        ["Dataset Insights", "What-If Simulator", "Predict from Manual Input", "Historical Reports"])

# --- HEADER ---
st.title("🏙️ AQI Prediction & Monitoring System")
st.write(f"**Data Status:** Dataset Driven | **Last Data Entry:** {df['Timestamp'].max().strftime('%d %B, %Y %I:%M %p')}")

# --- MAIN LOGIC ---

# 1. DATASET INSIGHTS (আগের Live Analytics এর পরিবর্তে)
if mode == "Dataset Insights":
    st.subheader("📋 Latest Statistics from Dataset")
    
    # ডাটাসেটের শেষ সারির ডেটা নেওয়া
    latest_data = df.iloc[-1]
    curr_aqi = latest_data['AQI']
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.plotly_chart(create_gauge(curr_aqi, "Latest AQI in Dataset"), use_container_width=True)
    with col2:
        st.metric("PM2.5 Level", f"{latest_data['PM2.5']}")
        st.metric("NO2 Level", f"{latest_data['NO2']}")
    with col3:
        st.metric("SO2 Level", f"{latest_data['SO2']}")
        st.metric("CO Level", f"{latest_data['CO']}")

    st.divider()
    
    # Charts Row
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 AQI Trend (Time Series)")
        fig_line = px.line(df, x='Timestamp', y='AQI', title="Historical AQI Trend")
        st.plotly_chart(fig_line, use_container_width=True)
    with c2:
        st.subheader("🍕 Pollutant Contribution (Pie Chart)")
        avg_pollutants = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].mean().reset_index()
        avg_pollutants.columns = ['Pollutant', 'Value']
        fig_pie = px.pie(avg_pollutants, values='Value', names='Pollutant', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# 2. WHAT-IF SIMULATOR
elif mode == "What-If Simulator":
    st.subheader("🧪 Machine Learning Scenario Simulator")
    st.write("গ্যাসের মান পরিবর্তন করে দেখুন প্রেডিকশনে কী প্রভাব পড়ে।")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s_pm2 = st.slider("PM2.5", 0, 500, 100)
        s_no2 = st.slider("NO2", 0, 200, 30)
    with col_s2:
        s_pm10 = st.slider("PM10", 0, 500, 120)
        s_co = st.slider("CO", 0.0, 10.0, 1.5)
    
    # Prediction logic (মডেলের ট্রেনিং ফিচার অর্ডার অনুযায়ী)
    now = datetime.now()
    features = np.array([[10, s_no2, s_co, 20, s_pm2, s_pm10, now.year, now.month, now.day, now.hour, now.weekday(), 0, s_pm2*1.1, s_pm2]])
    sim_res = model.predict(features)[0]
    
    st.plotly_chart(create_gauge(sim_res, "Simulated AQI Result"), use_container_width=True)
    st.info(f"এই সিমুলেশন অনুযায়ী প্রেডিক্টেড AQI হচ্ছে: **{sim_res:.2f}**")

# 3. PREDICT FROM MANUAL INPUT
elif mode == "Predict from Manual Input":
    st.subheader("⌨️ Manual Parameter Entry")
    
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        pm25 = c1.number_input("PM2.5", value=50.0)
        pm10 = c1.number_input("PM10", value=80.0)
        no2 = c2.number_input("NO2", value=20.0)
        so2 = c2.number_input("SO2", value=10.0)
        co = c3.number_input("CO", value=1.0)
        o3 = c3.number_input("O3", value=15.0)
        
        submit = st.form_submit_button("Predict AQI")
        
    if submit:
        now = datetime.now()
        features = np.array([[so2, no2, co, o3, pm25, pm10, now.year, now.month, now.day, now.hour, now.weekday(), 0, pm25*1.1, pm25]])
        res = model.predict(features)[0]
        
        st.success(f"### Predicted AQI: {res:.2f}")
        # Health Advice
        if res <= 100: st.write("✅ **Health Advice:** Air is fresh. Good for outdoor activities.")
        elif res <= 200: st.write("⚠️ **Health Advice:** Use a mask if you have respiratory issues.")
        else: st.write("🚫 **Health Advice:** Hazardous air! Stay indoors.")

# 4. HISTORICAL REPORTS
elif mode == "Historical Reports":
    st.subheader("📂 Dataset Explorer & Download")
    st.write("আপনার প্রি-প্রসেসড ডাটাসেটের সারাংশ নিচে দেওয়া হলো:")
    
    st.dataframe(df.describe(), use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full CSV Report", data=csv, file_name="aqi_data_report.csv", mime='text/csv')
    
    st.divider()
    st.subheader("🔍 Preview Top 20 Rows")
    st.table(df.head(20))

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Built for CSE Final Year Project by Naim. Dataset driven analysis.")
