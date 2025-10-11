import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px

# Load dataset
df = pd.read_csv("collected_data_cleaned.csv")
df['temperature_zscore'] = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
df['humidity_zscore'] = (df['humidity'] - df['humidity'].mean()) / df['humidity'].std()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.title("Filters")

# store city in session_state
if "city_selected" not in st.session_state:
    st.session_state.city_selected = ""

city_input = st.sidebar.text_input("Enter City Name")
search_button = st.sidebar.button("🔍 Search City")

if search_button and city_input.strip():
    st.session_state.city_selected = city_input.strip()

pollutants = st.sidebar.multiselect(
    "Select Pollutants",
    ["pm25", "pm10", "no2", "so2", "co", "o3"],
    default=["pm25"]
)
pm_threshold = st.sidebar.number_input("PM2.5 Alert Threshold", 0, 500, 100)

# -----------------------------
# Filter data based on saved city
# -----------------------------
filtered_df = pd.DataFrame()
if st.session_state.city_selected:
    filtered_df = df[df['city'].str.lower() == st.session_state.city_selected.lower()]

# -----------------------------
# Display selected city data
# -----------------------------
st.title(" Real-Time Pollution Dashboard")
if not filtered_df.empty:
    st.subheader(f"📊 Details for {st.session_state.city_selected}")
    st.table(filtered_df.T)  # transpose for better readability
else:
    st.info("Select a city to see its details.")

# -----------------------------
# Real-time alerts
# -----------------------------
st.subheader("⚠️ Alerts")
if not filtered_df.empty:
    if filtered_df.iloc[0]['pm25'] > pm_threshold:
        st.error(f"⚠️ High PM2.5 Alert for {st.session_state.city_selected}! Current value: {filtered_df.iloc[0]['pm25']}")
    else:
        st.success(f"No alerts for {st.session_state.city_selected}.")

# -----------------------------
# Map with circle marker
# -----------------------------
st.subheader("📍 Pollution Map")
if not filtered_df.empty:
    row = filtered_df.iloc[0]
    m = folium.Map(location=[row['latitude'], row['longitude']], zoom_start=6)

    # Determine color based on source
    if row['factories_count'] > 20:
        color = 'red'
        source_label = 'Industrial'
    elif row['roads_count'] > 5000:
        color = 'blue'
        source_label = 'Vehicular'
    else:
        color = 'green'
        source_label = 'Other'

    popup_text = (
        f"<b>City:</b> {row['city']}<br>"
        f"<b>Source:</b> {source_label}<br>"
        f"<b>PM2.5:</b> {row['pm25']}<br>"
        f"<b>PM10:</b> {row['pm10']}<br>"
        f"<b>NO2:</b> {row['no2']}<br>"
        f"<b>SO2:</b> {row['so2']}<br>"
        f"<b>CO:</b> {row['co']}<br>"
        f"<b>O3:</b> {row['o3']}<br>"
    )
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        popup=popup_text
    ).add_to(m)

    st_folium(m, width=700, height=500)

# -----------------------------
# Pie chart for selected pollutants
# -----------------------------
if not filtered_df.empty and pollutants:
    st.subheader(f"📊 Selected Pollutant Levels in {st.session_state.city_selected}")
    melted_df = filtered_df.melt(
        id_vars=["city"],
        value_vars=pollutants,
        var_name="Pollutant",
        value_name="Level"
    )
    fig = px.pie(
        melted_df,
        names="Pollutant",
        values="Level",
        title=f"Pollutant Distribution in {st.session_state.city_selected}",
        hole=0.3
    )
    st.plotly_chart(fig)

# -----------------------------
# Download CSV
# -----------------------------
st.subheader("💾 Download Report")
if not filtered_df.empty:
    st.download_button(
        label=f"Download CSV for {st.session_state.city_selected}",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name=f"pollution_report_{st.session_state.city_selected}.csv",
        mime="text/csv"
    )
else:
    st.info("Select a city to enable CSV download.")
