import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px

# --- CONFIGURATION ---
DATA_FILE = 'data/labeled_predictions.csv' # Now correctly loads from the main directory
MAP_CENTER = [20.0, 77.0] # Center of India (adjust to your project location)

# --- HELPER FUNCTIONS ---
def load_data():
    """Load the predicted data with location info."""
    try:
        # Assuming the final CSV has 'Latitude', 'Longitude', and 'Predicted_Source'
        df = pd.read_csv(DATA_FILE)
        # Ensure 'Pollution_Level' is numeric for heatmaps/alerts
        df['Pollution_Level'] = pd.to_numeric(df['Pollution_Level'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Check file path and column names.")
        return pd.DataFrame()

def get_color(source):
    """Assign a color based on the predicted source for markers."""
    colors = {'Industrial': 'red', 'Vehicular': 'blue', 'Natural': 'green', 'Other': 'gray'}
    return colors.get(source, 'gray')

# --- STREAMLIT DASHBOARD LAYOUT ---
st.set_page_config(layout="wide", page_title="Pollution Source Identifier Dashboard")

st.title("🏭 Pollution Source Identifier & Geospatial Dashboard")
st.markdown("---")

# Load Data
data = load_data()

if not data.empty:
    # --- SIDEBAR FILTERS (Module 6) ---
    st.sidebar.header("Data Filters")
    
    # 1. Location Input
    city_options = data['City'].unique() if 'City' in data.columns else ["Select Area"]
    selected_city = st.sidebar.selectbox("Select City/Area:", city_options)
    
    filtered_data = data[data['City'] == selected_city] if selected_city != "Select Area" and 'City' in data.columns else data

    # 2. Predicted Source Filter
    source_options = ['All'] + list(filtered_data['Predicted_Source'].unique())
    selected_source = st.sidebar.selectbox("Filter by Source:", source_options)

    if selected_source != 'All':
        filtered_data = filtered_data[filtered_data['Predicted_Source'] == selected_source]
        
    st.sidebar.markdown(f"**Data Points Displayed: {len(filtered_data)}**")
    st.sidebar.markdown("---")
    
    # --- VISUAL COMPONENTS (Module 5) ---
    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("Geospatial Map & Source Hotspots")

        # Create Folium Map
        m = folium.Map(location=MAP_CENTER, zoom_start=5, control_scale=True)
        
        # 1. Heatmap Overlay
        if 'Pollution_Level' in filtered_data.columns:
            # Heatmap data must be [lat, lon, value]
            heat_data = filtered_data[['Latitude', 'Longitude', 'Pollution_Level']].values.tolist()
            HeatMap(heat_data, min_opacity=0.2, max_val=filtered_data['Pollution_Level'].max(), radius=20).add_to(m)

        # 2. Source-Specific Markers
        marker_group = folium.FeatureGroup(name="Pollution Sources").add_to(m)
        for index, row in filtered_data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=get_color(row['Predicted_Source']),
                fill=True,
                fill_color=get_color(row['Predicted_Source']),
                tooltip=f"Source: {row['Predicted_Source']} | Conf: {row.get('Confidence_Score', 'N/A'):.2f} | Poll Level: {row.get('Pollution_Level', 'N/A')}",
            ).add_to(marker_group)
        
        # Display the map
        st_folium(m, width=900, height=500, key="folium_map", returned_objects={})

    with col2:
        st.header("Prediction Summary & Alerts")

        # 1. Real-time Alerts (Module 6)
        HIGH_THRESHOLD = 150 # Mock threshold for PM2.5/Pollution Index
        high_risk_data = filtered_data[filtered_data['Pollution_Level'] > HIGH_THRESHOLD]
        
        if not high_risk_data.empty:
            st.warning(f"🚨 **ALERT:** {len(high_risk_data)} locations exceed the safe threshold ({HIGH_THRESHOLD})!")
            st.markdown("**Top High-Risk Sites:**")
            st.dataframe(high_risk_data[['City', 'Pollution_Level', 'Predicted_Source', 'Confidence_Score']].head(), use_container_width=True)
        else:
            st.success("✅ Pollution levels are currently within safe limits in this area.")

        st.markdown("---")

        # 2. Pie Chart for Predicted Source Distribution (Module 6)
        source_counts = filtered_data['Predicted_Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        
        fig_pie = px.pie(source_counts, values='Count', names='Source', title='Predicted Source Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- USER INTERACTION FEATURES (Download Option - Module 6) ---
    st.markdown("---")
    
    # 3. Add a Download Button
    @st.cache_data
    def convert_df_to_csv(df):
        # Only include relevant columns for the report
        report_df = df[['Latitude', 'Longitude', 'City', 'Pollution_Level', 'Predicted_Source', 'Confidence_Score']]
        return report_df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_data)
    st.download_button(
        label="Download Daily Pollution Report (CSV)",
        data=csv,
        file_name='daily_pollution_report.csv',
        mime='text/csv',
    )
    
else:
    st.warning("Please ensure your data is processed and saved to the correct path.")