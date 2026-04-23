import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AQI Google Sheets Dashboard", layout="wide")

# --- LOAD ASSETS (Model & GSheets) ---
@st.cache_resource
def load_assets():
    # মডেল লোড
    model = joblib.load('aqi_xgboost_model.pkl')
    
    # গুগল শিট কানেকশন
    # এখানে আপনার গুগল শিটের 'Public' বা 'Share link' দিতে হবে
    sheet_url = "https://docs.google.com/spreadsheets/d/13GpYlkHVKrLb5vb-5PjfXFp_ceNoBqYP6QpMNOlOWgs/edit?usp=sharing"
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(spreadsheet=sheet_url)
    
    # ডেটা প্রি-প্রসেসিং (Timestamp ফিক্স করা)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    else:
        df['Timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1h')
    
    return model, df

# গুগল শিট ইউআরএল না বসানো পর্যন্ত এরর হ্যান্ডলিং
try:
    model, df = load_assets()
except Exception as e:
    st.error("গুগল শিট কানেক্ট করা যাচ্ছে না। দয়া করে সঠিক URL দিন।")
    st.stop()

# --- GAUGE METER FUNCTION ---
def create_gauge(value, title_text="Current AQI"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title_text},
        gauge = {
            'axis': {'range': [None, 500]},
            'steps': [
                {'range': [0, 50], 'color': "#00e400"},
                {'range': [51, 100], 'color': "#ffff00"},
                {'range': [101, 200], 'color': "#ff7e00"},
                {'range': [201, 300], 'color': "#ff0000"},
                {'range': [301, 500], 'color': "#8f3f97"}],
            'bar': {'color': "black"}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("📊 AQI Sheet Analytics")
mode = st.sidebar.radio("Features", ["Dataset Insights", "Scenario Simulator", "Manual Prediction", "Reports"])

# --- HEADER ---
st.title("🏙️ Smart AQI Dashboard (G-Sheets Integrated)")
st.write(f"**Connected to:** Google Sheets | **Last Entry:** {df['Timestamp'].max()}")

# ---------------- FEATURES ----------------

if mode == "Dataset Insights":
    # ১. ভিজ্যুয়াল অ্যানালিটিক্স
    latest_aqi = df.iloc[-1]['AQI']
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(create_gauge(latest_aqi), use_container_width=True)
    with col2:
        st.subheader("📈 AQI Time Series")
        fig_line = px.line(df, x='Timestamp', y='AQI', title="Historical Trend")
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()
    
    # Pollutant Breakdown (Pie Chart)
    st.subheader("🍕 Pollutant Contribution")
    avg_vals = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].mean().reset_index()
    avg_vals.columns = ['Pollutant', 'Value']
    st.plotly_chart(px.pie(avg_vals, values='Value', names='Pollutant', hole=0.4))

elif mode == "Scenario Simulator":
    # ২. স্মার্ট প্রেডিকশন (What-If)
    st.subheader("🧪 'What-If' Simulation")
    s_pm25 = st.slider("Adjust PM2.5 Level", 0, 500, 100)
    
    # মডেল প্রেডিকশন লজিক
    now = datetime.now()
    features = np.array([[15, 30, 1.2, 20, s_pm25, 120, now.year, now.month, now.day, now.hour, now.weekday(), 0, s_pm25*1.1, s_pm25]])
    res = model.predict(features)[0]
    
    st.plotly_chart(create_gauge(res, "Predicted Result"), use_container_width=True)
    st.info(f"PM2.5 লেভেল {s_pm25} হলে সম্ভাব্য AQI হবে {res:.2f}")

elif mode == "Manual Prediction":
    # ৩. স্বাস্থ্য ও সচেতনতা (Health Insights)
    st.subheader("⌨️ Manual Parameter Check")
    with st.form("manual_form"):
        p2 = st.number_input("PM2.5", value=60.0)
        p10 = st.number_input("PM10", value=110.0)
        submit = st.form_submit_button("Analyze")
        
    if submit:
        # প্রেডিকশন এবং হেলথ টিপস
        now = datetime.now()
        feat = np.array([[10, 25, 1.0, 15, p2, p10, now.year, now.month, now.day, now.hour, now.weekday(), 0, p2*1.1, p2]])
        pred = model.predict(feat)[0]
        
        st.metric("Predicted AQI", f"{pred:.2f}")
        if pred > 150:
            st.error("🚨 স্বাস্থ্য ঝুঁকি: বাইরে যাওয়া এড়িয়ে চলুন এবং মাস্ক পড়ুন।")
        else:
            st.success("✅ বায়ুমান সন্তোষজনক।")

elif mode == "Reports":
    # ৫. ডেটা ইঞ্জিনিয়ারিং (Download Reports)
    st.subheader("📂 Spreadsheet Data Explorer")
    st.dataframe(df.tail(20), use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sheet Data as CSV", data=csv, file_name="aqi_sheet_report.csv")

st.sidebar.markdown("---")
st.sidebar.caption("Connected via GSheets Connection v1.0")
