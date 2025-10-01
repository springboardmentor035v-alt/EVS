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
import random

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

# --- DATA GENERATION FUNCTIONS ---
def generate_extended_data(df):
    """Generate extended date range with realistic patterns"""
    extended_data = []
    
    # Get the original date from your data
    original_date = df['timestamp'].min()
    
    # Generate data for past 90 days and future 30 days
    for days_offset in range(-90, 31):  # -90 days to +30 days
        current_date = original_date + timedelta(days=days_offset)
        
        for _, original_row in df.iterrows():
            new_row = original_row.copy()
            
            # Modify timestamp
            new_row['timestamp'] = current_date.replace(
                hour=original_row['timestamp'].hour,
                minute=original_row['timestamp'].minute,
                second=original_row['timestamp'].second
            )
            
            # Add realistic variations to pollution data based on date patterns
            variation_factor = 1.0
            
            # Weekend effect (lower pollution on weekends)
            if current_date.weekday() >= 5:  # Saturday or Sunday
                variation_factor *= random.uniform(0.6, 0.8)  # 20-40% reduction on weekends
            else:
                variation_factor *= random.uniform(0.9, 1.2)  # Weekday variations
            
            # Seasonal patterns
            month = current_date.month
            if month in [11, 12, 1, 2]:  # Winter - higher pollution
                variation_factor *= random.uniform(1.1, 1.4)
            elif month in [6, 7, 8, 9]:  # Monsoon - lower pollution
                variation_factor *= random.uniform(0.7, 0.9)
            else:  # Spring/Autumn
                variation_factor *= random.uniform(0.9, 1.1)
            
            # Diurnal patterns (higher during rush hours)
            hour = original_row['timestamp'].hour
            if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                variation_factor *= random.uniform(1.2, 1.5)
            elif hour in [0, 1, 2, 3, 4, 5]:  # Early morning
                variation_factor *= random.uniform(0.6, 0.8)
            
            # Apply variations to pollutant concentrations
            pollutant_columns = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3', 'no']
            for col in pollutant_columns:
                if col in new_row and pd.notna(new_row[col]):
                    new_row[col] = max(0.1, new_row[col] * variation_factor * random.uniform(0.8, 1.2))
            
            # Vary pollution source occasionally (10% chance to change)
            if random.random() < 0.1:
                sources = ['Industrial', 'Vehicular', 'Agricultural Burning', 'Natural/Low', 'Mixed/Other', 'Construction Dust']
                new_row['pollution_source'] = random.choice(sources)
            
            extended_data.append(new_row)
    
    extended_df = pd.DataFrame(extended_data)
    
    # Recalculate AQI categories for new data
    extended_df['aqi_category'] = calculate_aqi_category(extended_df['pm2_5'])
    
    return extended_df

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

# --- DATA LOADING (Cached for performance) ---
@st.cache_data(ttl=3600)
def load_data():
    """Load and merge processed data with labels, then extend date range"""
    try:
        df_eda = pd.read_csv(PROCESSED_EDA_FILE, parse_dates=['timestamp'])
        df_labeled = pd.read_csv(LABELED_FILE)
        
        # Data validation
        if len(df_eda) != len(df_labeled):
            st.warning("Data length mismatch between EDA and labeled datasets")
            
        df_eda['pollution_source'] = df_labeled['pollution_source']
        
        # Generate extended data with more dates
        st.info("🔄 Generating extended date range data...")
        extended_df = generate_extended_data(df_eda)
        
        # Calculate additional metrics
        extended_df['aqi_category'] = calculate_aqi_category(extended_df['pm2_5'])
        extended_df['hour'] = extended_df['timestamp'].dt.hour
        extended_df['day_of_week'] = extended_df['timestamp'].dt.day_name()
        extended_df['date'] = extended_df['timestamp'].dt.date
        extended_df['month'] = extended_df['timestamp'].dt.month
        extended_df['season'] = extended_df['timestamp'].dt.month.map(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else 'Fall')
        extended_df['weekend'] = extended_df['timestamp'].dt.dayofweek >= 5
        
        st.success(f"✅ Generated data from {extended_df['timestamp'].min().strftime('%Y-%m-%d')} to {extended_df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        return extended_df
        
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

# Display data range info
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
total_days = (max_date - min_date).days + 1

st.sidebar.info(f"""
**📅 Data Range:**
- **From:** {min_date.strftime('%Y-%m-%d')}
- **To:** {max_date.strftime('%Y-%m-%d')}
- **Total Days:** {total_days}
- **Records:** {len(df):,}
""")

# --- SIDEBAR FILTERS ---
st.sidebar.header("📊 Dashboard Controls")

# City filter
selected_city = st.sidebar.selectbox(
    "Select City", 
    ['All Cities'] + sorted(df['name'].unique().tolist()),
    help="Filter data by specific urban area"
)

# Pollution source filter - only show sources that exist in data
available_sources = sorted(df['pollution_source'].unique().tolist())
selected_source = st.sidebar.selectbox(
    "Pollution Source", 
    ['All Sources'] + available_sources,
    help="Filter by predicted emission source type"
)

# Date range filter with extended range
st.sidebar.subheader("📅 Analysis Period")
date_preset = st.sidebar.radio(
    "Quick Date Ranges:",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Data", "Custom Range"],
    index=3  # Default to "All Data"
)

if date_preset == "Last 7 Days":
    date_range = (max_date - timedelta(days=7), max_date)
elif date_preset == "Last 30 Days":
    date_range = (max_date - timedelta(days=30), max_date)
elif date_preset == "Last 90 Days":
    date_range = (max_date - timedelta(days=90), max_date)
elif date_preset == "All Data":
    date_range = (min_date, max_date)
else:
    date_range = st.sidebar.date_input(
        "Custom Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
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

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]).replace(hour=23, minute=59, second=59)
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

# Calculate date range info for current selection
current_min_date = filtered_df['timestamp'].min().date()
current_max_date = filtered_df['timestamp'].max().date()
current_days = (current_max_date - current_min_date).days + 1

# Display metrics in columns
col1, col2, col3, col4, col5, col6 = st.columns(6)

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

with col6:
    st.metric(
        "Date Range", 
        f"{current_days} days",
        f"{current_min_date} to {current_max_date}",
        help="Analysis period coverage"
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
                'Mixed/Other': {'icon': 'info-circle', 'color': 'gray', 'prefix': 'fa'},
                'Construction Dust': {'icon': 'building', 'color': 'beige', 'prefix': 'fa'}
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
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <p>EnviroScan Analytics Dashboard v2.1 | Environmental Monitoring System</p>
    <p style='font-size: 0.8em;'>Data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} | {total_days} days | {len(df):,} records</p>
    <p style='font-size: 0.8em;'>Data last updated: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)