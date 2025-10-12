import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EnviroScan: Pollution Source Identifier",
    page_icon="💨",
    layout="wide"
)

# --- 2. CACHED DATA LOADING ---
@st.cache_data
def load_data():
    """
    Loads all necessary data and models.
    Uses caching to prevent reloading on every interaction.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, 'models', 'pollution_source_model.joblib')
    TRAIN_PATH = os.path.join(script_dir, 'data', 'train.csv')
    APP_DATA_PATH = os.path.join(script_dir, 'data', 'app_daily_data.csv')

    try:
        model = joblib.load(MODEL_PATH)
        train_df = pd.read_csv(TRAIN_PATH)
        app_df = pd.read_csv(APP_DATA_PATH, parse_dates=['timestamp'])
        app_df['timestamp'] = app_df['timestamp'].dt.tz_localize(None)
    except FileNotFoundError as e:
        st.error(f"ERROR: A required data file was not found: {e.filename}")
        st.error("Please run the scripts/preprocess_for_app.py script first to generate the app data.")
        st.stop()

    unique_labels = sorted(np.unique(train_df['pollution_source']))
    inverse_class_mapping = {i: label for i, label in enumerate(unique_labels)}

    numerical_cols = [
        'co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'temperature', 'humidity',
        'wind_speed', 'wind_direction', 'distance_to_nearest_industrial_m',
        'distance_to_nearest_major_roads_m', 'distance_to_nearest_dump_site_m',
        'distance_to_nearest_agricultural_m'
    ]
    
    scaler = StandardScaler()
    valid_cols_for_fit = [col for col in numerical_cols if col in train_df.columns]
    scaler.fit(train_df[valid_cols_for_fit])
    
    model_columns = train_df.drop('pollution_source', axis=1).columns

    return model, app_df, scaler, inverse_class_mapping, model_columns, numerical_cols

# --- 3. MAP GENERATION FUNCTION ---
def create_map(df):
    """Creates a Folium map with markers based on the input dataframe."""
    if df.empty:
        return folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    
    # Adjust zoom based on data spread
    lat_range = df['latitude'].max() - df['latitude'].min()
    lon_range = df['longitude'].max() - df['longitude'].min()
    
    if lat_range < 0.1 and lon_range < 0.1:
        zoom_start = 12  # Sub-area view
    elif lat_range < 0.5 and lon_range < 0.5:
        zoom_start = 10  # City view
    elif len(df['location_name'].unique()) == 1:
        zoom_start = 8
    else:
        zoom_start = 5
    
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")

    source_styles = {
        'Vehicular': {'color': 'blue', 'icon': 'car'},
        'Industrial': {'color': 'gray', 'icon': 'industry'},
        'Agricultural_Burning': {'color': 'orange', 'icon': 'fire'},
        'Background_Mixed': {'color': 'green', 'icon': 'leaf'}
    }
    
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df

    for _, row in df_sample.iterrows():
        source = row.get('predicted_source', 'N/A')
        confidence = row.get('confidence', 0)
        style = source_styles.get(source, {'color': 'black', 'icon': 'question-sign'})
        
        sub_area_info = f"<b>Sub-area:</b> {row.get('sub_area', 'N/A')}<br>" if 'sub_area' in row else ""
        
        popup_html = f"""
        <b>City:</b> {row['location_name']}<br>
        {sub_area_info}
        <b>Predicted Source:</b> {source}<br>
        <b>Confidence:</b> {confidence:.2%}<br>
        <hr>
        <b>PM2.5:</b> {row.get('pm25', 0):.2f} µg/m³<br>
        <b>PM10:</b> {row.get('pm10', 0):.2f} µg/m³<br>
        <b>NO2:</b> {row.get('no2', 0):.2f} µg/m³<br>
        <b>O3:</b> {row.get('o3', 0):.2f} µg/m³<br>
        <b>CO:</b> {row.get('co', 0):.2f} µg/m³<br>
        <b>Date:</b> {row['timestamp'].strftime('%Y-%m-%d')}
        """
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=350),
            icon=folium.Icon(color=style['color'], icon=style['icon'], prefix='fa')
        ).add_to(m)

    return m

# --- 4. ENHANCED PIE CHART WITH AGRICULTURAL DATA ---
def create_enhanced_pie_chart(filtered_df):
    """Creates a pie chart showing pollution sources and agricultural influence"""
    
    # Calculate source distribution
    source_counts = filtered_df['predicted_source'].value_counts()
    
    # Calculate agricultural influence
    if 'distance_to_nearest_agricultural_m' in filtered_df.columns:
        avg_agri_distance = filtered_df['distance_to_nearest_agricultural_m'].mean()
        near_agriculture = len(filtered_df[filtered_df['distance_to_nearest_agricultural_m'] < 2000])
        agri_percentage = (near_agriculture / len(filtered_df)) * 100
    else:
        avg_agri_distance = None
        agri_percentage = 0
    
    # Create the pie chart
    fig_pie = px.pie(
        values=source_counts.values, 
        names=source_counts.index, 
        title="Pollution Source Distribution",
        color_discrete_map={
            'Vehicular': '#3498db',
            'Industrial': '#95a5a6',
            'Agricultural_Burning': '#e67e22',
            'Background_Mixed': '#27ae60'
        }
    )
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig_pie, avg_agri_distance, agri_percentage

# --- 5. MAIN APPLICATION ---
model, app_df, scaler, inverse_class_mapping, model_columns, numerical_cols = load_data()

# Create sub_area column based on coordinates if not exists
if 'sub_area' not in app_df.columns:
    # Generate sub-areas based on coordinate clusters (simplified approach)
    app_df['sub_area'] = app_df.groupby('location_name').apply(
        lambda x: pd.cut(x['latitude'], bins=min(5, len(x.index)), labels=False, duplicates='drop')
    ).reset_index(level=0, drop=True)
    app_df['sub_area'] = app_df['sub_area'].apply(lambda x: f"Area {int(x)+1}" if pd.notna(x) else "Area 1")

# --- SIDEBAR FILTERS ---
st.sidebar.title("🎯 Filters")

# City selection
if 'location_name' not in app_df.columns:
    st.error("The 'location_name' column is missing from app_daily_data.csv.")
    st.error("Please re-run the scripts/preprocess_for_app.py script to fix the data file.")
    st.stop()

city_list = ["All Cities"] + sorted(app_df['location_name'].unique().tolist())
selected_city = st.sidebar.selectbox("🏙 Select City", options=city_list)

# Sub-area selection (appears only when a specific city is selected)
selected_sub_area = "All Sub-areas"
if selected_city != "All Cities":
    city_data = app_df[app_df['location_name'] == selected_city]
    sub_area_list = ["All Sub-areas"] + sorted(city_data['sub_area'].unique().tolist())
    selected_sub_area = st.sidebar.selectbox("📍 Select Sub-area", options=sub_area_list)

# Date range selection
min_date = app_df['timestamp'].min().date()
max_date = app_df['timestamp'].max().date()

date_range = st.sidebar.date_input(
    "📅 Select Date Range",
    value=(max_date - timedelta(days=30), max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD"
)

# Pollutant filter
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Pollutant Display")
show_pollutants = st.sidebar.multiselect(
    "Select pollutants to display",
    options=['pm25', 'pm10', 'no2', 'o3', 'co', 'so2'],
    default=['pm25', 'pm10', 'no2', 'o3', 'co']
)

# --- DATA FILTERING LOGIC ---
filtered_df = app_df.copy()

if selected_city != "All Cities":
    filtered_df = filtered_df[filtered_df['location_name'] == selected_city]
    
    if selected_sub_area != "All Sub-areas":
        filtered_df = filtered_df[filtered_df['sub_area'] == selected_sub_area]

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered_df = filtered_df[
        (filtered_df['timestamp'].dt.date >= start_date.date()) & 
        (filtered_df['timestamp'].dt.date <= end_date.date())
    ]

# --- MAIN PAGE LAYOUT ---
st.title("🌍 EnviroScan: AI-Powered Pollution Source Identifier")
st.markdown("This dashboard visualizes daily average pollution data and predicts the likely source using a machine learning model.")

# Display current selection
location_text = selected_city
if selected_city != "All Cities" and selected_sub_area != "All Sub-areas":
    location_text = f"{selected_city} - {selected_sub_area}"

st.info(f"📊 Analyzing data for: *{location_text}* | Date Range: *{date_range[0]}* to *{date_range[1]}*")

# --- PREDICTIONS ---
if not filtered_df.empty:
    df_to_predict = filtered_df.copy()

    df_to_predict['is_weekend'] = (df_to_predict['timestamp'].dt.dayofweek >= 5).astype(int)
    df_to_predict['month_sin'] = np.sin(2 * np.pi * df_to_predict['timestamp'].dt.month / 12.0)
    df_to_predict['month_cos'] = np.cos(2 * np.pi * df_to_predict['timestamp'].dt.month / 12.0)
    
    df_to_predict = pd.get_dummies(df_to_predict, columns=['location_name'], prefix='location')
    df_to_predict = df_to_predict.reindex(columns=model_columns, fill_value=0)

    valid_numerical_cols = [col for col in numerical_cols if col in df_to_predict.columns]
    
    if not df_to_predict.empty:
        df_to_predict[valid_numerical_cols] = scaler.transform(df_to_predict[valid_numerical_cols])

        predictions = model.predict(df_to_predict)
        probabilities = model.predict_proba(df_to_predict)
        
        filtered_df['predicted_source'] = pd.Series(predictions, index=filtered_df.index).map(inverse_class_mapping)
        filtered_df['confidence'] = probabilities.max(axis=1)
        
        # --- KEY METRICS ---
        st.header(f"📈 Analysis for: {location_text}")
        latest_data = filtered_df.sort_values('timestamp').iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Predicted Source", latest_data['predicted_source'])
        col2.metric("Model Confidence", f"{latest_data['confidence']:.2%}")
        col3.metric("PM2.5 Level", f"{latest_data.get('pm25', 0):.2f} µg/m³")
        col4.metric("PM10 Level", f"{latest_data.get('pm10', 0):.2f} µg/m³")

        # Air quality alerts
        SAFE_PM25_THRESHOLD = 60
        SAFE_PM10_THRESHOLD = 100
        
        if latest_data.get('pm25', 0) > SAFE_PM25_THRESHOLD or latest_data.get('pm10', 0) > SAFE_PM10_THRESHOLD:
            st.error(f"🚨 AIR QUALITY ALERT: PM2.5 ({latest_data.get('pm25', 0):.2f}) or PM10 ({latest_data.get('pm10', 0):.2f}) exceeds safe thresholds!", icon="🚨")

        # --- VISUALIZATION SECTION ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🥧 Source Distribution & Agricultural Impact")
            
            # Enhanced pie chart
            fig_pie, avg_agri_distance, agri_percentage = create_enhanced_pie_chart(filtered_df)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Agricultural impact metrics
            st.markdown("### 🌾 Agricultural Influence")
            if avg_agri_distance is not None:
                st.metric("Avg. Distance to Agriculture", f"{avg_agri_distance:.0f} m")
                st.metric("Locations Near Agriculture (<2km)", f"{agri_percentage:.1f}%")
                
                if agri_percentage > 50:
                    st.warning("⚠ High agricultural influence detected")
            
            # Download button
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df_to_csv(filtered_df)
            st.download_button(
                label="📥 Download Report as CSV",
                data=csv,
                file_name=f"{selected_city.replace(' ', '_')}_pollution_report.csv",
                mime='text/csv',
            )

        with col2:
            st.subheader("📊 Multi-Pollutant Trends Over Time")
            
            # Prepare data for plotting
            if selected_city != "All Cities" and selected_sub_area == "All Sub-areas":
                plot_df = filtered_df
                title_text = f"Daily Average Pollutant Levels in {selected_city}"
            elif selected_city != "All Cities" and selected_sub_area != "All Sub-areas":
                plot_df = filtered_df
                title_text = f"Pollutant Levels in {selected_city} - {selected_sub_area}"
            else:
                plot_df = filtered_df.groupby('timestamp').mean(numeric_only=True).reset_index()
                title_text = "Daily Average Pollutant Levels (All Cities)"
            
            # Filter pollutants to display based on user selection
            available_pollutants = [p for p in show_pollutants if p in plot_df.columns]
            
            if available_pollutants:
                fig_line = px.line(
                    plot_df, 
                    x='timestamp', 
                    y=available_pollutants, 
                    title=title_text,
                    labels={'value': 'Concentration (µg/m³)', 'timestamp': 'Date'},
                    color_discrete_map={
                        'pm25': '#e74c3c',
                        'pm10': '#e67e22',
                        'no2': '#3498db',
                        'o3': '#9b59b6',
                        'co': '#2ecc71',
                        'so2': '#f39c12'
                    }
                )
                fig_line.update_layout(hovermode='x unified')
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning("No pollutant data available for the selected filters.")

        # --- ADDITIONAL POLLUTANT COMPARISON ---
        st.subheader("🔬 Pollutant Concentration Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PM levels comparison
            if 'pm25' in filtered_df.columns and 'pm10' in filtered_df.columns:
                pm_data = pd.DataFrame({
                    'Pollutant': ['PM2.5', 'PM10'],
                    'Average': [filtered_df['pm25'].mean(), filtered_df['pm10'].mean()],
                    'Max': [filtered_df['pm25'].max(), filtered_df['pm10'].max()]
                })
                fig_pm = px.bar(pm_data, x='Pollutant', y=['Average', 'Max'], 
                               title="PM Levels", barmode='group')
                st.plotly_chart(fig_pm, use_container_width=True)
        
        with col2:
            # Gaseous pollutants
            if all(p in filtered_df.columns for p in ['no2', 'o3', 'co']):
                gas_data = pd.DataFrame({
                    'Pollutant': ['NO2', 'O3', 'CO'],
                    'Average': [
                        filtered_df['no2'].mean(), 
                        filtered_df['o3'].mean(), 
                        filtered_df['co'].mean()
                    ]
                })
                fig_gas = px.bar(gas_data, x='Pollutant', y='Average',
                                title="Gaseous Pollutants (Avg)", color='Pollutant')
                st.plotly_chart(fig_gas, use_container_width=True)
        
        with col3:
            # Correlation heatmap
            pollutant_cols = [p for p in ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2'] 
                            if p in filtered_df.columns]
            if len(pollutant_cols) > 1:
                corr_matrix = filtered_df[pollutant_cols].corr()
                fig_heatmap = px.imshow(
                    corr_matrix,
                    title="Pollutant Correlations",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

        # --- GEOSPATIAL MAP ---
        st.subheader("🗺 Geospatial Analysis Map")
        map_to_display = create_map(filtered_df)
        st_folium(map_to_display, width='100%', height=500, returned_objects=[])

        # --- DETAILED STATISTICS TABLE ---
        with st.expander("📋 View Detailed Statistics"):
            stats_df = filtered_df[['timestamp', 'location_name', 'sub_area', 'predicted_source', 
                                    'confidence'] + available_pollutants].tail(50)
            st.dataframe(stats_df, use_container_width=True)

    else:
        st.warning("⚠ No data available after preprocessing. Please check your data.")
else:
    st.warning("⚠ No data available for the selected filters. Please adjust your date range, city, or sub-area selection.")

# --- FOOTER ---
st.markdown("---")
st.markdown("Developed with ❤ using Streamlit | Data updated daily")
