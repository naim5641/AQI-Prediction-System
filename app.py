import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart AQI Forecast", page_icon="🌍", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # নিশ্চিত করুন আপনার গিটহাবে এই নামেই মডেলটি আছে
        return joblib.load('aqi_xgboost_model.pkl')
    except:
        return None

model = load_model()

# --- WAQI API FUNCTION ---
def get_live_data(city):
    # আপনার দেওয়া টোকেন এখানে বসানো হয়েছে
    API_TOKEN = "8336110b0e0345c68d0dc9ca4554b7122e5ad33d"
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == 'ok':
            iaqi = data['data']['iaqi']
            
            # API থেকে তথ্য সংগ্রহ (তথ্য না থাকলে ডিফল্ট মান ব্যবহার করা হবে)
            pm25 = iaqi.get('pm25', {}).get('v', 50.0)
            pm10 = iaqi.get('pm10', {}).get('v', 100.0)
            no2  = iaqi.get('no2', {}).get('v', 30.0)
            so2  = iaqi.get('so2', {}).get('v', 10.0)
            co   = iaqi.get('co', {}).get('v', 1.0)
            o3   = iaqi.get('o3', {}).get('v', 20.0)
            
            return pm25, pm10, no2, so2, co, o3, True
        else:
            return 0,0,0,0,0,0, False
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return 0,0,0,0,0,0, False

# --- UI DESIGN ---
st.title("🏙️ Real-Time AQI Prediction System")
st.markdown(f"**Target Location:** Bangladesh | **Current Local Time:** {datetime.now().strftime('%I:%M %p')}")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Prediction", "Project Details"])

if page == "Live Prediction":
    st.subheader("📍 Select a City for Real-Time Analysis")
    # বাংলাদেশের প্রধান শহরগুলো
    selected_city = st.selectbox("Choose City", ["Dhaka", "Chittagong", "Cumilla", "Sylhet", "Rajshahi", "Khulna"])

    if st.button("🚀 Fetch Live Data & Predict"):
        with st.spinner(f'Connecting to WAQI sensors in {selected_city}...'):
            pm25, pm10, no2, so2, co, o3, success = get_live_data(selected_city)
            
            if success:
                now = datetime.now()
                # মডেলের ট্রেনিং সিকোয়েন্স অনুযায়ী ইনপুট সাজানো
                # ['SO2', 'NO2', 'CO', 'O3', 'PM2.5', 'PM10', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend', 'AQI_Moving_Avg', 'AQI_Lag_1']
                input_features = np.array([[
                    so2, no2, co, o3, pm25, pm10, 
                    now.year, now.month, now.day, now.hour, 
                    now.weekday(), (1 if now.weekday() >= 5 else 0), 
                    pm25*1.15, pm25*1.05 # Lag features placeholder
                ]])
                
                if model:
                    prediction = model.predict(input_features)[0]
                    
                    # Layout for Results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.info(f"### Current Sensor Data ({selected_city})")
                        st.write(f"🧪 **PM2.5:** {pm25} µg/m³")
                        st.write(f"🧪 **PM10:** {pm10} µg/m³")
                        st.write(f"🧪 **NO2:** {no2} ppb")
                    
                    with col2:
                        st.success(f"### Predicted AQI: {prediction:.2f}")
                        # Health Advice Logic
                        if prediction <= 50:
                            st.markdown("✅ **Status: Good** - বাতাসের মান ভালো। বাইরে ব্যায়াম করতে পারেন।")
                        elif prediction <= 100:
                            st.markdown("⚠️ **Status: Moderate** - অতি সংবেদনশীল মানুষরা সাবধানে থাকুন।")
                        elif prediction <= 200:
                            st.markdown("🔴 **Status: Unhealthy** - মাস্ক ব্যবহার করুন এবং দীর্ঘক্ষণ বাইরে থাকা এড়িয়ে চলুন।")
                        else:
                            st.markdown("💀 **Status: Hazardous** - বিপদজনক পরিস্থিতি! ঘরে অবস্থান করুন।")
                    
                    st.balloons()
                else:
                    st.error("Model file 'aqi_xgboost_model.pkl' not found in the repository.")
            else:
                st.error("Could not fetch data for this city. Please try again later.")

else:
    st.subheader("About the Project")
    st.write("This is an End-to-End Machine Learning project developed for AQI monitoring in Bangladesh.")
    st.markdown("""
    - **Model:** XGBoost Regressor
    - **Data Source:** World Air Quality Index (WAQI) API
    - **Tech Stack:** Python, Streamlit, Scikit-learn, Joblib
    """)

st.divider()
st.caption("CSE Final Year Project | Built by Naim | Supervised by Mentors")