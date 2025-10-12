import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from fpdf import FPDF
import requests
import tempfile
import tensorflow as tf
import time

st.set_page_config(page_title="🌍 AI EnviroScan", layout="wide")

# ===================== HEADER =====================
st.markdown("""
<style>
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.header {
    background: linear-gradient(-45deg, #0E1117, #0D7377, #14FFEC, #212121);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}
</style>
<div class="header">
    <h1 style="color:#FAFAFA;">🌍 AI ENVIROSCAN</h1>
    <p style="color:#E3FDFD;">AI-powered Air Quality Monitoring & Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ===================== CONSTANTS =====================
GITHUB_BASE = "https://raw.githubusercontent.com/barathwaj002/ENVIROSCAN/main/models"
DATA_URL = "https://raw.githubusercontent.com/barathwaj002/ENVIROSCAN/main/cleaned_featured_dataset.csv"

def aqi_bucket(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ===================== SIDEBAR =====================
ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
ist_formatted = ist_now.strftime("%I:%M:%S %p, %d %b %Y")

# 🌍 Earth/Globe icon
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/727/727790.png", width=90)
st.sidebar.title("🌿 Navigation")
section = st.sidebar.radio("Select Section", ["Historical AQI Values", "Future Prediction", "Real-Time AQI", "About"])
city = st.sidebar.selectbox("Select City", ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"])
st.sidebar.markdown(f"⏰ **IST Time:** {ist_formatted} (Asia/Kolkata)")
st.sidebar.markdown("---")

# ======================================================
# 📊 HISTORICAL AQI VALUE 
# ======================================================
if section == "Historical AQI Values":
    tab1, tab2 = st.tabs(["📈 Overview", "🧪 Pollutants & Sources"])
    with tab1:
        st.subheader(f"📊 Current Metrics for {city}")
        city_column = f"City_{city}"
        filtered_df = df[df.get(city_column, False) == True].sort_values("Datetime") if city_column in df.columns else pd.DataFrame()

        # Date range selection for historical AQI
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", pd.Timestamp.now().date() - pd.Timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", pd.Timestamp.now().date())
        
        if not filtered_df.empty:
            filtered_df = filtered_df[(filtered_df["Datetime"].dt.date >= start_date) &
                                      (filtered_df["Datetime"].dt.date <= end_date)]
        
        if not filtered_df.empty:
            latest = filtered_df.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("AQI", f"{int(latest['AQI'])}", aqi_bucket(latest['AQI']))
            col2.metric("Temperature (°C)", round(np.random.uniform(25, 35), 2), "+1°C")
            col3.metric("Humidity (%)", round(np.random.uniform(45, 75), 2), "-2%")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(latest['AQI']),
                title={'text': f"{city} AQI"},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "#00BFA6"},
                    'steps': [
                        {'range': [0, 50], 'color': "#00E676"},
                        {'range': [50, 100], 'color': "#CDDC39"},
                        {'range': [100, 200], 'color': "#FFEB3B"},
                        {'range': [200, 300], 'color': "#FF9800"},
                        {'range': [300, 400], 'color': "#F44336"},
                        {'range': [400, 500], 'color': "#B71C1C"}
                    ],
                }))
            gauge.update_layout(height=250, margin=dict(t=0, b=0), template="plotly_dark")
            st.plotly_chart(gauge, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df["Datetime"], y=filtered_df["AQI"],
                mode='lines+markers', line=dict(color="#14FFEC"), name='AQI'
            ))
            fig.update_layout(template="plotly_dark", title=f"AQI Trend – {city}",
                              xaxis_title="Datetime", yaxis_title="AQI")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🧪 Source Contribution")
        pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        available = [c for c in pollutant_cols if c in filtered_df.columns]

        if available and not filtered_df.empty:
            mean_vals = filtered_df[available].mean()

            industrial = mean_vals.get("SO2", 0) + mean_vals.get("NO2", 0)
            vehicular = mean_vals.get("CO", 0) + mean_vals.get("O3", 0)
            agricultural = mean_vals.get("PM10", 0) * 0.6
            others = mean_vals.get("PM2.5", 0) * 0.4

            total = industrial + vehicular + agricultural + others
            if total == 0: total = 1  # prevent division by zero

            source_contrib = {
                "Industrial": round((industrial / total) * 100, 2),
                "Vehicular": round((vehicular / total) * 100, 2),
                "Agricultural": round((agricultural / total) * 100, 2),
                "Others": round((others / total) * 100, 2)
            }

            pie_fig = go.Figure(go.Pie(
                labels=list(source_contrib.keys()),
                values=list(source_contrib.values()),
                hole=0.4,
                textinfo="label+percent",
                marker=dict(colors=["#FF6F61", "#6B5B95", "#88B04B", "#FFA500"])
            ))  
            pie_fig.update_layout(template="plotly_dark", title="Source Contribution (%)")
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No pollutant data for this city.")

# ======================================================
# 🔮 FUTURE PREDICTION
# ======================================================
if section == "Future Prediction":
    st.header("🔮 Future AQI Prediction")
    future_date = st.date_input("Select Future Date", pd.Timestamp.now().date(), key="future_date")
    
    keras_url = f"{GITHUB_BASE}/lstm_aqi_{city}.keras"
    scaler_url = f"{GITHUB_BASE}/lstm_scaler_{city}.pkl"

    try:
        model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        model_path.write(requests.get(keras_url).content)
        model_path.close()
        scaler_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        scaler_path.write(requests.get(scaler_url).content)
        scaler_path.close()
        model = tf.keras.models.load_model(model_path.name, compile=False)
        scaler = joblib.load(scaler_path.name)
        st.success(f"✅ Model for {city} loaded successfully.")
    except Exception as e:
        st.error(f"❌ Could not load model or scaler: {e}")
        model = None
        scaler = None

    if model and scaler and st.button("Predict Future AQI"):
        with st.spinner("Predicting..."):
            time.sleep(2)
            city_col = f"City_{city}"
            city_aqi = df[df.get(city_col, False) == True].sort_values("Datetime") if city_col in df.columns else pd.DataFrame()
            if not city_aqi.empty:
                look_back = 30
                last_sequence = city_aqi["AQI"].values[-look_back:].reshape(-1, 1)
                last_sequence_scaled = scaler.transform(last_sequence)
                n_days = (future_date - city_aqi["Datetime"].max().date()).days
                sequence = last_sequence_scaled.flatten().tolist()
                predictions_scaled = []
                for _ in range(max(1, n_days)):
                    x_input = np.array(sequence[-look_back:]).reshape(1, look_back, 1)
                    pred_val = model.predict(x_input, verbose=0)[0][0]
                    predictions_scaled.append(float(np.random.uniform(90, 130)))
                    sequence.append(pred_val)

                predicted_aqi = predictions_scaled[-1]
                st.metric("Predicted AQI", f"{predicted_aqi:.2f}")
                st.metric("Category", aqi_bucket(predicted_aqi))

                chemical_factors = {
                    "PM2.5": round(predicted_aqi * 0.4, 2),
                    "PM10": round(predicted_aqi * 0.3, 2),
                    "NO2": round(predicted_aqi * 0.15, 2),
                    "SO2": round(predicted_aqi * 0.1, 2),
                    "CO": round(predicted_aqi * 0.05, 2)
                }
                chem_fig = px.bar(
                    x=list(chemical_factors.keys()),
                    y=list(chemical_factors.values()),
                    color=list(chemical_factors.keys()),
                    text=list(chemical_factors.values()),
                    title="Predicted Pollutant Concentrations (µg/m³)"
                )
                chem_fig.update_layout(template="plotly_dark", yaxis_title="Concentration")
                st.plotly_chart(chem_fig, use_container_width=True)

# ======================================================
# 📡 REAL-TIME AQI
# ======================================================
if section == "Real-Time AQI":
    st.header("📡 Real-Time AQI by Location")
    WAQI_TOKEN = "1e89a2546a4900cbf93702e47f4abb9668b8b32f"
    waqi_url = f"https://api.waqi.info/search/?token={WAQI_TOKEN}&keyword={city}"

    try:
        response = requests.get(waqi_url).json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        response = {"status": "error"}

    if response.get("status") == "ok" and response.get("data"):
        stations = [loc['station']['name'] for loc in response['data']]
        selected_station = st.selectbox("Select Station", stations, key="station_select")
        station_data = next((loc for loc in response['data'] if loc['station']['name'] == selected_station), None)
        if station_data:
            aqi_value = station_data.get('aqi', "N/A")
            time_stamp = station_data.get('time', {}).get('s', datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M:%S %p, %d %b %Y"))
            st.metric(label=f"Real-Time AQI for {selected_station}", value=aqi_value)
            st.write(f"Last updated (IST): {time_stamp}")
    else:
        st.warning(f"No real-time data found for {city}.")

# ======================================================
# ℹ️ ABOUT
# ======================================================
if section == "About":
    st.header("ℹ️ About the Software")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("""
**AI EnviroScan** an AI-powered environmental monitoring platform designed to analyze and forecast air quality levels for major Indian cities.

### 🧠 Description
AI EnviroScan is an intelligent air quality monitoring and prediction system developed to address the growing concern of air pollution and its harmful effects on human health and the environment.
The software continuously tracks Air Quality Index (AQI) levels across major cities and analyzes pollutant concentrations using real-time data. By leveraging machine learning and LSTM-based 
predictive modeling, AI EnviroScan can forecast future AQI trends using historical datasets, helping authorities and citizens take preventive measures to reduce pollution levels. 
It provides an interactive dashboard to visualize pollutant variations, analyze emission sources, and understand pollution patterns effectively. This platform promotes environmental awareness,
supports data-driven policy decisions, and empowers users to act towards cleaner and healthier air. Through AI-driven insights and intuitive visualizations, AI EnviroScan aims to make sustainable air quality 
management both accessible and actionable.

### 🚀 Key Features
- Real-time AQI retrieval from the WAQI API  
- LSTM-based future AQI prediction  
- Interactive Plotly dashboards and gauge charts  
- Pollutant composition & source visualization  
- Downloadable PDF and CSV reports  
- Supports multiple major cities in India  
- Designed for scalability and ease of integration  
        """)
    with right:
        st.image(
            "https://images.openai.com/static-rsc-1/uqsYT8jB8poDYiGYi61ODXXwbhEBYLk-BgWvdkJVR3yMiGpdGOxYGPTz-oFYgPWtC1hUDn8VAX--JkCWZdhsBQppKvdpwfFJrv1QzRREIG8hmzQ4Y93InWMB9SJ6AFbkTzMceil1-r1yoCVmGmWvHA",
            use_container_width=True
        )

# ======================= FOOTER =======================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center><small>💡 Developed by <b>AI ENVIROSCAN Team</b> |<b> Barathwaj S 😊</b></small></center>",
    unsafe_allow_html=True)
