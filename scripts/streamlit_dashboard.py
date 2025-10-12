import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
from io import BytesIO

# ----------------- PAGE CONFIG -----------------
st.set_page_config(layout="wide", page_title="Pollution Source Dashboard")
st.title("🌍 Pollution Source Dashboard")

# ----------------- LOAD DATA -----------------
df = pd.read_csv("data/labeled_data_new.csv")
df = df.dropna(subset=["latitude", "longitude", "aqi"])  # ensure no NaNs

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("Filters")

# Pollution Sources
sources = df["pollution_source"].unique().tolist()
selected_sources = st.sidebar.multiselect(
    "Select Pollution Sources",
    options=sources,
    default=sources,
    key="pollution_sources"
)

# Pollutants
pollutants = ["pm2_5","pm10","no2","so2","o3","co","aqi"]
selected_pollutant = st.sidebar.selectbox("Select Pollutant for Heatmap/Charts", pollutants)

# Month filter for trend chart
months = sorted(df["month"].unique())
selected_months = st.sidebar.multiselect("Select Month(s)", months, default=months)

# ----------------- FILTER DATA -----------------
df_filtered_sources = df[df["pollution_source"].isin(selected_sources)]
df_filtered_trend = df_filtered_sources[df_filtered_sources["month"].isin(selected_months)]

st.write(f"Displaying {df_filtered_sources.shape[0]} locations for Pollution Sources Map")

# ----------------- POLLUTION SOURCES MAP -----------------
m_sources = folium.Map(
    location=[df_filtered_sources["latitude"].mean(), df_filtered_sources["longitude"].mean()],
    zoom_start=4
)

# Source colors
source_colors = {
    "Vehicular": "red",
    "Industrial": "green",
    "Agricultural": "orange",
    "Burning": "purple",
    "Natural": "blue"
}

# Add all filtered points
for _, row in df_filtered_sources.iterrows():
    folium.Circle(
        location=[row["latitude"], row["longitude"]],
        radius=2000,
        color=source_colors.get(row["pollution_source"], "gray"),
        fill=True,
        fill_opacity=0.6,
        popup=f"Location ID: {row['location_id']}<br>"
              f"Source: {row['pollution_source']}"
    ).add_to(m_sources)

# Add legend for sources
legend_html_sources = """
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 150px; height: 140px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;">
&emsp;<b>Pollution Sources</b><br>
&emsp;<i style="color:red;">●</i> Vehicular<br>
&emsp;<i style="color:green;">●</i> Industrial<br>
&emsp;<i style="color:orange;">●</i> Agricultural<br>
&emsp;<i style="color:purple;">●</i> Burning<br>
&emsp;<i style="color:blue;">●</i> Natural
</div>
"""
m_sources.get_root().html.add_child(folium.Element(legend_html_sources))

st.subheader("Pollution Sources Map")
st_folium(m_sources, width=1200, height=700)



# ----------------- HEATMAP -----------------
st.subheader(f"🔥 {selected_pollutant} Heatmap")
m_heat = folium.Map(location=[df_filtered_sources["latitude"].mean(), df_filtered_sources["longitude"].mean()], zoom_start=4)
heat_data = [[row['latitude'], row['longitude'], row[selected_pollutant]] for _, row in df_filtered_sources.iterrows()]
HeatMap(heat_data, radius=15, max_zoom=13).add_to(m_heat)
st_folium(m_heat, width=1200, height=700)

# ----------------- TREND CHART -----------------
st.subheader(f"📈 {selected_pollutant} Trend by Month")
trend_df = df_filtered_trend.groupby("month")[selected_pollutant].mean().reset_index()
fig_trend = px.line(trend_df, x="month", y=selected_pollutant, markers=True, title=f"{selected_pollutant} Trend")
st.plotly_chart(fig_trend, use_container_width=True)



# ----------------- PIE CHART -----------------
st.subheader("🟢 Predicted Pollution Source Distribution")
pie_df = df_filtered_sources["pollution_source"].value_counts().reset_index()
pie_df.columns = ["pollution_source", "count"]
fig_pie = px.pie(pie_df, names="pollution_source", values="count", color="pollution_source",
                 color_discrete_map=source_colors, title="Pollution Source Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# ----------------- ALERTS -----------------
st.subheader("⚠️ High-Risk Areas (AQI)")
aqi_threshold = 1.0  # adjust according to scaled data
high_risk = df_filtered_sources[df_filtered_sources["aqi"] > aqi_threshold]
if not high_risk.empty:
    st.warning(f"{high_risk.shape[0]} locations exceed safe AQI thresholds!")
    st.dataframe(high_risk[["location_id","locality","country","pollution_source","aqi"]])
else:
    st.success("All locations are within safe AQI levels.")

# ----------------- POLLUTANT MAP WITH MARKER CLUSTER AND LEGEND -----------------
st.subheader("🌡️ Pollution Map by Selected Pollutant")

df_filtered_pollutant = df_filtered_sources.dropna(subset=["latitude", "longitude"])

pollutants = ["pm2_5","pm10","no2","so2","o3","co","aqi","temperature","humidity","wind_speed","temp_humidity_index","pollution_wind_ratio"]
selected_pollutant = st.sidebar.selectbox("Select Pollutant to Visualize", pollutants, key="pollutant_map")

m_pollutant = folium.Map(
    location=[df_filtered_pollutant["latitude"].mean(), df_filtered_pollutant["longitude"].mean()],
    zoom_start=4
)

marker_cluster = MarkerCluster().add_to(m_pollutant)

min_val = df_filtered_pollutant[selected_pollutant].min()
max_val = df_filtered_pollutant[selected_pollutant].max()
range_val = max_val - min_val if max_val - min_val != 0 else 1

def pollutant_color(val):
    norm = (val - min_val) / range_val
    if norm < 0.33:
        return "green"
    elif norm < 0.66:
        return "orange"
    else:
        return "red"

for _, row in df_filtered_pollutant.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=6,
        color=pollutant_color(row[selected_pollutant]),
        fill=True,
        fill_opacity=0.7,
        popup=(f"Location ID: {row['location_id']}<br>"
               f"Locality: {row['locality']}<br>"
               f"Country: {row['country']}<br>"
               f"{selected_pollutant}: {row[selected_pollutant]:.2f}<br>"
               f"Source: {row['pollution_source']}")
    ).add_to(marker_cluster)

legend_html_pollutant = f"""
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 150px; height: 100px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;">
&emsp;<b>{selected_pollutant} Intensity</b><br>
&emsp;<i style="color:green;">●</i> Low (< {min_val + range_val*0.33:.2f})<br>
&emsp;<i style="color:orange;">●</i> Medium (< {min_val + range_val*0.66:.2f})<br>
&emsp;<i style="color:red;">●</i> High (≥ {min_val + range_val*0.66:.2f})
</div>
"""
m_pollutant.get_root().html.add_child(folium.Element(legend_html_pollutant))

st_folium(m_pollutant, width=1200, height=700)

# ----------------- AQI HIGH-RISK HEATMAP -----------------
st.subheader("🔥 AQI High-Risk Heatmap")
df_aqi = df_filtered_sources.dropna(subset=["latitude", "longitude", "aqi"])
heat_data = [[row["latitude"], row["longitude"], row["aqi"]] for _, row in df_aqi.iterrows()]
m_aqi = folium.Map(
    location=[df_aqi["latitude"].mean(), df_aqi["longitude"].mean()],
    zoom_start=4
)
HeatMap(heat_data, min_opacity=0.4, max_opacity=0.8, radius=25, blur=15,
        gradient={0.2: 'green', 0.5: 'orange', 0.8: 'red'}).add_to(m_aqi)
legend_html_aqi = """
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 150px; height: 100px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white;">
&emsp;<b>AQI Severity</b><br>
&emsp;<i style="color:green;">●</i> Low<br>
&emsp;<i style="color:orange;">●</i> Medium<br>
&emsp;<i style="color:red;">●</i> High
</div>
"""
m_aqi.get_root().html.add_child(folium.Element(legend_html_aqi))
st_folium(m_aqi, width=1200, height=700)

# ----------------- DOWNLOAD OPTION -----------------
st.subheader("📥 Download Pollution Report")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_filtered_sources)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='pollution_report.csv',
    mime='text/csv',
)
