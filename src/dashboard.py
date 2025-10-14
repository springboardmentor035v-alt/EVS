import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import pdfkit
import folium
from streamlit_folium import st_folium
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(layout="wide", page_title="🌍 EnviroScan Dashboard")

# --- Custom Dark Theme ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .css-1d391kg { background-color: #161a23; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; }
    .stTabs [aria-selected="true"] { background-color: #1f2937; color: #00FFFF; }
    h1, h2, h3, h4, h5 { color: #00FFFF; }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/final_predictions.csv")
        df.dropna(subset=['latitude', 'longitude', 'city'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found: data/final_predictions.csv")
        st.info("Please run your data pipeline to generate it.")
        return None

df = load_data()
if df is None:
    st.stop()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔧 Control Panel")
sources = sorted(df["predicted_source"].unique().tolist())
selected_sources = st.sidebar.multiselect("Pollution Sources", sources, default=sources)
cities = sorted(df["city"].unique().tolist())
selected_cities = st.sidebar.multiselect("Cities", cities, default=cities[:5])

df_filtered = df[df["predicted_source"].isin(selected_sources) & df["city"].isin(selected_cities)]

# -------------------- DASHBOARD TITLE --------------------
st.title("🌍 EnviroScan: AI-Powered Pollution Source Identifier")

# -------------------- METRICS --------------------
col1, col2, col3, col4 = st.columns(4)
avg_aqi = df_filtered['aqi'].mean()
high_risk = len(df_filtered[df_filtered["aqi"] > 3.0])
dominant_source = df_filtered["predicted_source"].mode()[0] if not df_filtered.empty else "N/A"

col1.metric("Total Locations", f"{df_filtered.shape[0]}")
col2.metric("Average AQI", f"{avg_aqi:.2f}")
col3.metric("High Risk Areas", f"{high_risk}")
col4.metric("Dominant Source", dominant_source)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map", "📊 Analytics", "⚠️ Alerts", "🤖 Predict"])

# -------------------- MAP TAB --------------------
with tab1:
    st.subheader("🗺️ Pollution Sources Map")
    if df_filtered.empty:
        st.warning("No data to display for the selected filters.")
    else:
        m = folium.Map(location=[df_filtered["latitude"].mean(), df_filtered["longitude"].mean()],
                       zoom_start=6, tiles="CartoDB dark_matter")

        colors = {'Vehicular': 'blue', 'Industrial': 'red', 'Burning': 'orange', 'Dust': 'beige', 'Other': 'gray'}
        for _, row in df_filtered.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=6,
                popup=f"<b>{row['name']}</b><br>{row['city']}<br>Source: {row['predicted_source']}<br>AQI: {row['aqi']:.2f}",
                color=colors.get(row["predicted_source"], "gray"),
                fill=True, fill_opacity=0.8
            ).add_to(m)
        st_folium(m, height=500, use_container_width=True)

# -------------------- ANALYTICS TAB --------------------
with tab2:
    st.subheader("📊 Source Analytics")
    if df_filtered.empty:
        st.warning("No data available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            source_counts = df_filtered["predicted_source"].value_counts()
            st.plotly_chart(px.pie(source_counts, values=source_counts.values,
                                   names=source_counts.index,
                                   title="Source Distribution",
                                   color_discrete_sequence=px.colors.sequential.Tealgrn),
                            use_container_width=True)
        with col2:
            avg_aqi_src = df_filtered.groupby("predicted_source")["aqi"].mean().sort_values()
            st.plotly_chart(px.bar(avg_aqi_src, x=avg_aqi_src.index, y='aqi',
                                   text_auto='.2f', title="Average AQI by Source",
                                   color=avg_aqi_src.index,
                                   color_discrete_sequence=px.colors.qualitative.Vivid),
                            use_container_width=True)

# -------------------- ALERTS TAB --------------------
with tab3:
    st.subheader("⚠️ High-Risk Pollution Alerts")
    high_risk_df = df_filtered[df_filtered["aqi"] > 3.0]
    if not high_risk_df.empty:
        st.error(f"{len(high_risk_df)} areas exceed safe AQI limits!")
        st.dataframe(high_risk_df[['city', 'name', 'predicted_source', 'aqi']])
    else:
        st.success("✅ All monitored areas are within safe AQI levels.")

# -------------------- PREDICT TAB --------------------
with tab4:
    st.subheader("🤖 Predict Pollution Source (What-If Analysis)")

    try:
        model = joblib.load("pollution_source_model.joblib")
        scaler = joblib.load("data_scaler.joblib")
    except FileNotFoundError:
        st.error("⚠️ Model files missing. Please run training first.")
        st.stop()

    features = [
        'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi',
        'latitude', 'longitude', 'dist_to_road_m', 'dist_to_industrial_m'
    ]

    col_pred1, col_pred2 = st.columns(2)
    inputs = {}

    with col_pred1:
        st.markdown("##### Pollutant & AQI Values")
        inputs['pm2_5'] = st.number_input("PM2.5", value=55.0, format="%.2f")
        inputs['pm10'] = st.number_input("PM10", value=90.0, format="%.2f")
        inputs['no2'] = st.number_input("NO2", value=45.0, format="%.2f")
        inputs['so2'] = st.number_input("SO2", value=20.0, format="%.2f")
        inputs['o3'] = st.number_input("O3", value=35.0, format="%.2f")
        inputs['co'] = st.number_input("CO", value=1.2, format="%.2f")
        inputs['aqi'] = st.number_input("Air Quality Index (AQI)", value=3.0, format="%.2f")

    with col_pred2:
        st.markdown("##### Geospatial & Distance Features")
        inputs['latitude'] = st.number_input("Latitude", value=28.61, format="%.4f")
        inputs['longitude'] = st.number_input("Longitude", value=77.23, format="%.4f")
        inputs['dist_to_road_m'] = st.number_input("Distance to Nearest Road (m)", value=200.0, format="%.1f")
        inputs['dist_to_industrial_m'] = st.number_input("Distance to Industrial Zone (m)", value=1000.0, format="%.1f")

    if st.button("🔮 Predict Source", use_container_width=True):
        input_df = pd.DataFrame([inputs])[features]
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]

        st.success(f"### Predicted Source: **{pred}**")

        # --- Confidence Visualization ---
        prob_df = pd.DataFrame({
            'Source': model.classes_,
            'Confidence (%)': probs * 100
        }).sort_values(by='Confidence (%)', ascending=False)

        fig_conf = px.bar(prob_df, x='Source', y='Confidence (%)',
                          text_auto='.1f', title="Prediction Confidence by Source",
                          color='Source', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_conf, use_container_width=True)

        # --- PDF REPORT EXPORT ---
        if st.button("📄 Generate PDF Report"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            html = f"""
            <h1 style='color:#00FFFF;'>🌍 EnviroScan Prediction Report</h1>
            <p><b>Generated:</b> {timestamp}</p>
            <h2 style='color:#00FFFF;'>Predicted Source: {pred}</h2>
            <p><b>Input Values:</b></p>
            {input_df.to_html(index=False)}
            {px.pie(prob_df, names='Source', values='Confidence (%)', title='Source Probability Distribution').to_html(full_html=False, include_plotlyjs='cdn')}
            {px.bar(prob_df, x='Source', y='Confidence (%)', title='Confidence Scores by Source').to_html(full_html=False, include_plotlyjs='cdn')}
            """
            pdfkit.from_string(html, "prediction_report.pdf")
            with open("prediction_report.pdf", "rb") as f:
                st.download_button("⬇️ Download Report (PDF)", f, file_name="prediction_report.pdf")
