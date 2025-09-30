import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EnviroScan Analytics Dashboard", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FILE PATHS ---
OUTPUT_DIR = "outputs"
PROCESSED_EDA_FILE = os.path.join(OUTPUT_DIR, "processed_data_for_eda.csv")
LABELED_FILE = os.path.join(OUTPUT_DIR, "labeled_data_for_dashboard.csv")
ENCODER_FILE = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

# --- DATA LOADING (Cached for performance) ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load and merge processed data with labels"""
    try:
        df_eda = pd.read_csv(PROCESSED_EDA_FILE, parse_dates=['timestamp'])
        df_labeled = pd.read_csv(LABELED_FILE)
        
        # Data validation
        if len(df_eda) != len(df_labeled):
            st.warning("Data length mismatch between EDA and labeled datasets")
            
        df_eda['pollution_source'] = df_labeled['pollution_source']
        
        # Calculate additional metrics
        df_eda['aqi_category'] = calculate_aqi_category(df_eda['pm2_5'])
        df_eda['hour'] = df_eda['timestamp'].dt.hour
        df_eda['day_of_week'] = df_eda['timestamp'].dt.day_name()
        df_eda['date'] = df_eda['timestamp'].dt.date
        
        return df_eda
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_encoder():
    """Load the label encoder for pollution sources"""
    try:
        return joblib.load(ENCODER_FILE)
    except FileNotFoundError:
        st.error("Label encoder file not found")
        return None

def calculate_aqi_category(pm25_values):
    """Calculate AQI categories based on PM2.5 levels"""
    categories = []
    for pm25 in pm25_values:
        if pm25 <= 12:
            categories.append("Good")
        elif pm25 <= 35.4:
            categories.append("Moderate")
        elif pm25 <= 55.4:
            categories.append("Unhealthy for Sensitive Groups")
        elif pm25 <= 150.4:
            categories.append("Unhealthy")
        elif pm25 <= 250.4:
            categories.append("Very Unhealthy")
        else:
            categories.append("Hazardous")
    return categories

# --- MAIN APP LOGIC ---
st.title("🌍 EnviroScan Environmental Analytics Dashboard")
st.markdown("""
    *Comprehensive analysis of air pollution patterns, sources, and trends across monitored urban environments.*
""")

# Load data
df = load_data()
encoder = load_encoder()

if df is None or encoder is None:
    st.error("""
    ❌ **Data initialization failed.** 
    
    Please ensure:
    1. The data pipeline has been executed using `python3 run_pipeline.py`
    2. All required data files are present in the `outputs/` directory
    3. File permissions are correctly set
    """)
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("📊 Dashboard Controls")

# City filter
selected_city = st.sidebar.selectbox(
    "Select City", 
    ['All Cities'] + sorted(df['name'].unique().tolist()),
    help="Filter data by specific urban area"
)

# Pollution source filter
sources = ['All Sources'] + list(encoder.classes_)
selected_source = st.sidebar.selectbox(
    "Pollution Source", 
    sources,
    help="Filter by predicted emission source type"
)

# Date range filter with smart defaults
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
default_end = max_date
default_start = max(default_end - timedelta(days=30), min_date)

selected_date_range = st.sidebar.date_input(
    "Analysis Period",
    value=(default_start, default_end),
    min_value=min_date,
    max_value=max_date,
    help="Select start and end dates for temporal analysis"
)

# AQI severity filter
aqi_levels = ['All Levels', 'Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
selected_aqi = st.sidebar.selectbox(
    "Air Quality Level",
    aqi_levels,
    help="Filter by air quality index severity"
)

# --- DATA FILTERING ---
filtered_df = df.copy()

# Apply filters
if selected_city != 'All Cities':
    filtered_df = filtered_df[filtered_df['name'] == selected_city]

if selected_source != 'All Sources':
    filtered_df = filtered_df[filtered_df['pollution_source'] == selected_source]

if len(selected_date_range) == 2:
    start_date = pd.to_datetime(selected_date_range[0])
    end_date = pd.to_datetime(selected_date_range[1]).replace(hour=23, minute=59, second=59)
    filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & (filtered_df['timestamp'] <= end_date)]

if selected_aqi != 'All Levels':
    filtered_df = filtered_df[filtered_df['aqi_category'] == selected_aqi]

# Handle empty results
if filtered_df.empty:
    st.warning("""
    🚫 **No data available for selected filters**
    
    Suggestions:
    - Broaden date range
    - Include more cities
    - Relax AQI severity filters
    """)
    st.stop()

# --- EXECUTIVE SUMMARY DASHBOARD ---
st.subheader("📈 Executive Summary")

# Calculate key metrics
avg_pm25 = filtered_df['pm2_5'].mean()
peak_pm25 = filtered_df['pm2_5'].max()
dominant_source = filtered_df['pollution_source'].mode()[0] if not filtered_df['pollution_source'].mode().empty else "Insufficient Data"
total_readings = len(filtered_df)
cities_covered = filtered_df['name'].nunique()
worst_aqi_percentage = (filtered_df['aqi_category'].isin(['Unhealthy', 'Very Unhealthy', 'Hazardous']).sum() / len(filtered_df)) * 100

# Display metrics in columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Average PM₂.₅", 
        f"{avg_pm25:.1f} µg/m³",
        help="Mean particulate matter concentration"
    )

with col2:
    st.metric(
        "Peak PM₂.₅", 
        f"{peak_pm25:.1f} µg/m³",
        help="Maximum recorded particulate matter level"
    )

with col3:
    st.metric(
        "Dominant Source", 
        dominant_source,
        help="Most frequently identified pollution source"
    )

with col4:
    st.metric(
        "Data Points", 
        f"{total_readings:,}",
        help="Total measurements in current selection"
    )

with col5:
    st.metric(
        "Poor Air Quality", 
        f"{worst_aqi_percentage:.1f}%",
        help="Percentage of readings in unhealthy ranges",
        delta=f"-{worst_aqi_percentage:.1f}%" if worst_aqi_percentage < 20 else None,
        delta_color="inverse"
    )

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Geospatial Analysis", 
    "📊 Temporal Trends", 
    "🔍 Source Analytics",
    "📋 Data Explorer"
])

with tab1:
    st.header("Geospatial Pollution Distribution")
    
    col_map, col_stats = st.columns([2, 1])
    
    with col_map:
        with st.spinner("Generating interactive map..."):
            # Dynamic map center calculation
            map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
            m = folium.Map(
                location=map_center, 
                zoom_start=10 if selected_city == 'All Cities' else 12,
                tiles="CartoDB positron"
            )

            # Heatmap layer
            heatmap_data = filtered_df[['latitude', 'longitude', 'pm2_5']].dropna()
            if not heatmap_data.empty:
                HeatMap(
                    data=heatmap_data.values.tolist(),
                    radius=20,
                    blur=18,
                    gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'},
                    min_opacity=0.3
                ).add_to(folium.FeatureGroup(name="PM₂.₅ Concentration Heatmap").add_to(m))

            # Source markers with enhanced styling
            source_icons = {
                'Industrial': {'icon': 'industry', 'color': 'darkred', 'prefix': 'fa'},
                'Vehicular': {'icon': 'car', 'color': 'blue', 'prefix': 'fa'},
                'Agricultural Burning': {'icon': 'fire', 'color': 'orange', 'prefix': 'fa'},
                'Natural/Low': {'icon': 'leaf', 'color': 'green', 'prefix': 'fa'},
                'Mixed/Other': {'icon': 'info-circle', 'color': 'gray', 'prefix': 'fa'}
            }
            
            mc = MarkerCluster(
                name="Pollution Source Locations",
                options={
                    'maxClusterRadius': 50,
                    'spiderfyOnMaxZoom': True,
                    'showCoverageOnHover': True
                }
            ).add_to(m)
            
            for _, row in filtered_df.iterrows():
                source = row['pollution_source']
                details = source_icons.get(source, source_icons['Mixed/Other'])
                
                # Enhanced popup with more details
                popup_html = f"""
                <div style="min-width: 200px;">
                    <h4>{row['name']}</h4>
                    <p><b>Time:</b> {row['timestamp'].strftime('%Y-%m-%d %H:%M')}</p>
                    <p><b>Source:</b> {source}</p>
                    <p><b>PM₂.₅:</b> {row['pm2_5']:.2f} µg/m³</p>
                    <p><b>AQI:</b> {row['aqi_category']}</p>
                    <p><b>NO₂:</b> {row.get('no2', 'N/A')}</p>
                    <p><b>SO₂:</b> {row.get('so2', 'N/A')}</p>
                </div>
                """
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{source}: {row['pm2_5']:.1f} µg/m³",
                    icon=folium.Icon(
                        color=details['color'], 
                        icon=details['icon'], 
                        prefix=details['prefix'],
                        icon_color='white'
                    )
                ).add_to(mc)
            
            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, use_container_width=True, height=500)
    
    with col_stats:
        st.subheader("Spatial Statistics")
        
        # City ranking by pollution
        city_stats = filtered_df.groupby('name').agg({
            'pm2_5': ['mean', 'max', 'count'],
            'pollution_source': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }).round(2)
        
        city_stats.columns = ['Avg PM₂.₅', 'Peak PM₂.₅', 'Readings', 'Dominant Source']
        city_stats = city_stats.sort_values('Avg PM₂.₅', ascending=False)
        
        st.dataframe(
            city_stats,
            use_container_width=True,
            height=300
        )
        
        # AQI distribution
        aqi_dist = filtered_df['aqi_category'].value_counts()
        fig_aqi_pie = px.pie(
            values=aqi_dist.values,
            names=aqi_dist.index,
            title="Air Quality Distribution",
            hole=0.4
        )
        fig_aqi_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_aqi_pie, use_container_width=True)

with tab2:
    st.header("Temporal Analysis & Trends")
    
    # Time series analysis
    col_ts1, col_ts2 = st.columns(2)
    
    with col_ts1:
        st.subheader("Pollutant Concentration Trends")
        
        # Multi-pollutant time series
        pollutants = ['pm2_5', 'no2', 'so2', 'o3', 'co']
        pollutant_names = ['PM₂.₅', 'NO₂', 'SO₂', 'O₃', 'CO']
        
        selected_pollutants = st.multiselect(
            "Select pollutants to display:",
            options=pollutants,
            default=['pm2_5', 'no2'],
            format_func=lambda x: dict(zip(pollutants, pollutant_names))[x]
        )
        
        if selected_pollutants:
            # Resample to daily averages for clarity
            daily_avg = filtered_df.set_index('timestamp').resample('D')[selected_pollutants].mean()
            
            fig_trend = go.Figure()
            for pollutant in selected_pollutants:
                fig_trend.add_trace(go.Scatter(
                    x=daily_avg.index,
                    y=daily_avg[pollutant],
                    name=dict(zip(pollutants, pollutant_names))[pollutant],
                    mode='lines+markers',
                    line=dict(width=2)
                ))
            
            fig_trend.update_layout(
                title="Daily Average Pollutant Concentrations",
                xaxis_title="Date",
                yaxis_title="Concentration (µg/m³)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_ts2:
        st.subheader("Diurnal Patterns")
        
        # Hourly patterns
        hourly_pattern = filtered_df.groupby('hour').agg({
            'pm2_5': 'mean',
            'no2': 'mean',
            'so2': 'mean'
        }).reset_index()
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=hourly_pattern['hour'],
            y=hourly_pattern['pm2_5'],
            name='PM₂.₅',
            line=dict(color='red', width=3)
        ))
        fig_hourly.add_trace(go.Scatter(
            x=hourly_pattern['hour'],
            y=hourly_pattern['no2'],
            name='NO₂',
            line=dict(color='blue', width=2)
        ))
        fig_hourly.add_trace(go.Scatter(
            x=hourly_pattern['hour'],
            y=hourly_pattern['so2'],
            name='SO₂',
            line=dict(color='orange', width=2)
        ))
        
        fig_hourly.update_layout(
            title="Average Hourly Pollutant Concentrations",
            xaxis_title="Hour of Day",
            yaxis_title="Concentration (µg/m³)",
            xaxis=dict(tickmode='linear', dtick=2),
            height=400
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Weekly and monthly patterns
    col_wk, col_src = st.columns(2)
    
    with col_wk:
        st.subheader("Weekly Patterns")
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = filtered_df.groupby('day_of_week')['pm2_5'].mean().reindex(day_order)
        
        fig_weekly = px.bar(
            weekly_avg, 
            y='pm2_5',
            title="Average PM₂.₅ by Day of Week",
            labels={'pm2_5': 'PM₂.₅ (µg/m³)', 'day_of_week': 'Day'}
        )
        fig_weekly.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col_src:
        st.subheader("Source Distribution Over Time")
        
        # Time-based source distribution
        source_trend = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'pollution_source']).size().unstack(fill_value=0)
        
        fig_source_trend = px.area(
            source_trend,
            title="Pollution Source Trends Over Time",
            labels={'value': 'Number of Readings', 'timestamp': 'Date'}
        )
        fig_source_trend.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_source_trend, use_container_width=True)

with tab3:
    st.header("Pollution Source Analytics")
    
    col_src1, col_src2 = st.columns(2)
    
    with col_src1:
        st.subheader("Source Contribution Analysis")
        
        # Source impact analysis
        source_impact = filtered_df.groupby('pollution_source').agg({
            'pm2_5': ['mean', 'max', 'count'],
            'no2': 'mean',
            'so2': 'mean'
        }).round(2)
        
        source_impact.columns = ['Avg PM₂.₅', 'Max PM₂.₅', 'Readings', 'Avg NO₂', 'Avg SO₂']
        source_impact = source_impact.sort_values('Avg PM₂.₅', ascending=False)
        
        st.dataframe(
            source_impact,
            use_container_width=True,
            height=400
        )
    
    with col_src2:
        st.subheader("Source Distribution")
        
        fig_source_pie = px.pie(
            filtered_df, 
            names='pollution_source', 
            title='Pollution Source Distribution',
            hole=0.4
        )
        fig_source_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            pull=[0.1 if i == 0 else 0 for i in range(len(filtered_df['pollution_source'].unique()))]
        )
        st.plotly_chart(fig_source_pie, use_container_width=True)
        
        # Source by city heatmap
        st.subheader("Source-City Matrix")
        source_city_matrix = pd.crosstab(filtered_df['name'], filtered_df['pollution_source'])
        fig_heatmap = px.imshow(
            source_city_matrix,
            title="Pollution Sources by City",
            aspect="auto",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    st.header("Advanced Data Exploration")
    
    # Top pollution events
    st.subheader("Significant Pollution Events")
    
    col_top1, col_top2 = st.columns([1, 2])
    
    with col_top1:
        top_n = st.slider(
            "Number of top events to display", 
            min_value=5, 
            max_value=100, 
            value=15,
            help="Select the number of highest pollution readings to analyze"
        )
        
        severity_threshold = st.slider(
            "PM₂.₅ threshold (µg/m³)",
            min_value=0,
            max_value=int(filtered_df['pm2_5'].max()),
            value=50,
            help="Filter events above specific concentration level"
        )
    
    with col_top2:
        severe_events = filtered_df[filtered_df['pm2_5'] >= severity_threshold].sort_values('pm2_5', ascending=False).head(top_n)
        
        if not severe_events.empty:
            st.dataframe(
                severe_events[[
                    'timestamp', 'name', 'pollution_source', 'pm2_5', 'aqi_category', 'no2', 'so2'
                ]].round(2),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No events meet the selected severity threshold.")
    
    # Statistical summary
    st.subheader("Statistical Summary")
    
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    stats_summary = filtered_df[numeric_cols].describe().round(3)
    
    st.dataframe(
        stats_summary,
        use_container_width=True
    )
    
    # Data download section
    st.subheader("Data Export")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.info("""
        **Export Options:**
        - Full filtered dataset (CSV)
        - Statistical summary (CSV)
        - Top pollution events (CSV)
        """)
    
    with col_dl2:
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')
        
        # Full dataset download
        csv_full = convert_df_to_csv(filtered_df)
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv_full,
            file_name=f'enviroscan_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
            help="Download complete filtered dataset"
        )
        
        # Top events download
        if not severe_events.empty:
            csv_events = convert_df_to_csv(severe_events)
            st.download_button(
                label="📥 Download Top Events (CSV)",
                data=csv_events,
                file_name=f'enviroscan_top_events_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
                help="Download significant pollution events only"
            )

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>EnviroScan Analytics Dashboard v2.1 | Environmental Monitoring System</p>
    <p style='font-size: 0.8em;'>Data last updated: {}</p>
</div>
""".format(df['timestamp'].max().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)