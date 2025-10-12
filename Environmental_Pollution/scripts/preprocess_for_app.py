# scripts/preprocess_for_app.py

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

print("Starting pre-processing for the Streamlit app...")

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(script_dir, '..', 'data', 'consolidated_enviro_data.csv')
OUTPUT_PATH = os.path.join(script_dir, '..', 'data', 'app_daily_data.csv')

# Use memory-efficient dtypes
dtype_spec = {
    'latitude': 'float32', 'longitude': 'float32', 'value': 'float32',
    'temperature': 'float32', 'humidity': 'float32', 'wind_speed': 'float32',
    'wind_direction': 'float32', 'distance_to_nearest_industrial_m': 'float32',
    'distance_to_nearest_major_roads_m': 'float32', 'distance_to_nearest_dump_site_m': 'float32',
    'distance_to_nearest_agricultural_m': 'float32'
}

def generate_sub_areas(city_df, n_clusters=5):
    """
    Generate sub-areas for a city based on geographical clustering.
    Uses K-means clustering on latitude and longitude coordinates.
    """
    if len(city_df) < n_clusters:
        n_clusters = max(1, len(city_df) // 10)
    
    if n_clusters <= 1:
        city_df['sub_area'] = 'Central Area'
        return city_df
    
    # Extract coordinates
    coords = city_df[['latitude', 'longitude']].drop_duplicates()
    
    if len(coords) < n_clusters:
        n_clusters = len(coords)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Create a mapping dataframe
    coords['cluster'] = kmeans.fit_predict(coords[['latitude', 'longitude']])
    
    # Name clusters based on geographical position
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['lat', 'lon'])
    cluster_centers['cluster_id'] = range(len(cluster_centers))
    
    # Determine cardinal direction for each cluster
    city_center_lat = city_df['latitude'].mean()
    city_center_lon = city_df['longitude'].mean()
    
    def get_direction_name(row, center_lat, center_lon):
        lat_diff = row['lat'] - center_lat
        lon_diff = row['lon'] - center_lon
        
        # Determine primary direction
        if abs(lat_diff) > abs(lon_diff):
            direction = "North" if lat_diff > 0 else "South"
        else:
            direction = "East" if lon_diff > 0 else "West"
        
        # Add secondary direction if significant
        if abs(lat_diff) > 0.01 and abs(lon_diff) > 0.01:
            secondary = "East" if lon_diff > 0 else "West"
            if lat_diff > 0:
                direction = f"North{secondary}"
            else:
                direction = f"South{secondary}"
        
        return direction
    
    cluster_centers['direction'] = cluster_centers.apply(
        lambda row: get_direction_name(row, city_center_lat, city_center_lon), 
        axis=1
    )
    
    # Create sub-area names
    cluster_centers['sub_area'] = cluster_centers['direction'] + ' Zone'
    
    # If there's only one zone, call it Central Area
    if len(cluster_centers) == 1:
        cluster_centers['sub_area'] = 'Central Area'
    
    # Merge cluster names back to original dataframe
    coords = coords.merge(
        cluster_centers[['cluster_id', 'sub_area']], 
        left_on='cluster', 
        right_on='cluster_id', 
        how='left'
    )
    
    # Merge with city_df
    city_df = city_df.merge(
        coords[['latitude', 'longitude', 'sub_area']], 
        on=['latitude', 'longitude'], 
        how='left'
    )
    
    # Fill any NaN values
    city_df['sub_area'] = city_df['sub_area'].fillna('Central Area')
    
    return city_df

try:
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=['timestamp'], dtype=dtype_spec)

    print("Pivoting data...")
    # Pivot to get pollutants as columns
    index_cols = ['location_name', 'latitude', 'longitude', 'timestamp']
    pivoted_df = df.set_index(index_cols + ['pollutant'])['value'].unstack(level='pollutant').reset_index()
    pivoted_df.columns.name = None

    print("Merging metadata...")
    # Get metadata columns (excluding the pollutants and value)
    metadata_cols = [col for col in df.columns if col not in ['pollutant', 'value', 'unit']]
    metadata = df[metadata_cols].drop_duplicates(subset=['location_name', 'timestamp'])
    
    # Merge back
    merged_df = pd.merge(pivoted_df, metadata, on=['location_name', 'latitude', 'longitude', 'timestamp'], how='left')

    print("Aggregating to daily averages...")
    # Set timestamp as index for resampling
    merged_df.set_index('timestamp', inplace=True)

    # Define columns
    numeric_cols_to_agg = [c for c in ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'temperature', 'humidity', 'wind_speed', 'wind_direction'] if c in merged_df.columns]
    static_cols = ['latitude', 'longitude', 'distance_to_nearest_industrial_m', 'distance_to_nearest_major_roads_m', 'distance_to_nearest_dump_site_m', 'distance_to_nearest_agricultural_m']
    valid_static_cols = [col for col in static_cols if col in merged_df.columns]

    # Group by location and resample
    grouped = merged_df.groupby('location_name').resample('D')
    
    # Aggregate
    agg_dict = {col: 'mean' for col in numeric_cols_to_agg}
    agg_dict.update({col: 'first' for col in valid_static_cols})
    daily_df = grouped.agg(agg_dict)

    # Fill gaps while still grouped
    daily_df = daily_df.ffill().bfill()

    # Reset index to make location_name and timestamp into columns
    daily_df = daily_df.reset_index()

    print("Generating sub-areas for each city...")
    # Generate sub-areas for each city
    sub_area_dfs = []
    for city in daily_df['location_name'].unique():
        print(f"  Processing sub-areas for {city}...")
        city_df = daily_df[daily_df['location_name'] == city].copy()
        city_df_with_subareas = generate_sub_areas(city_df, n_clusters=5)
        sub_area_dfs.append(city_df_with_subareas)
    
    # Combine all cities back together
    daily_df = pd.concat(sub_area_dfs, ignore_index=True)

    # Ensure all required pollutants are present
    required_pollutants = ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2']
    for pollutant in required_pollutants:
        if pollutant not in daily_df.columns:
            print(f"  Warning: {pollutant} not found in data, adding with NaN values")
            daily_df[pollutant] = np.nan

    print(f"Saving pre-processed data to {OUTPUT_PATH}...")
    
    # Verify location_name exists before saving
    if 'location_name' not in daily_df.columns:
        raise KeyError("Failed to create 'location_name' column during preprocessing.")
    
    # Verify sub_area exists
    if 'sub_area' not in daily_df.columns:
        raise KeyError("Failed to create 'sub_area' column during preprocessing.")
        
    daily_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Pre-processing complete!")
    print(f"   Total rows: {len(daily_df)}")
    print(f"   Cities: {daily_df['location_name'].nunique()}")
    print(f"   Sub-areas: {daily_df['sub_area'].nunique()}")
    print(f"   Date range: {daily_df['timestamp'].min()} to {daily_df['timestamp'].max()}")
    print(f"   Columns: {daily_df.columns.tolist()}")
    
    # Display sub-area distribution
    print("\n📊 Sub-area Distribution by City:")
    for city in sorted(daily_df['location_name'].unique()):
        city_data = daily_df[daily_df['location_name'] == city]
        sub_areas = city_data['sub_area'].unique()
        print(f"   {city}: {len(sub_areas)} sub-areas - {', '.join(sorted(sub_areas))}")
    
    # Display pollutant statistics
    print("\n🔬 Pollutant Data Summary:")
    for pollutant in required_pollutants:
        if pollutant in daily_df.columns:
            non_null = daily_df[pollutant].notna().sum()
            if non_null > 0:
                avg_val = daily_df[pollutant].mean()
                print(f"   {pollutant.upper()}: {non_null} records, avg: {avg_val:.2f} µg/m³")
            else:
                print(f"   {pollutant.upper()}: No data available")

except FileNotFoundError:
    print(f"❌ ERROR: Raw data file not found at {RAW_DATA_PATH}")
    print("   Please run the data collection script first (scripts/data_collector.py)")
except KeyError as e:
    print(f"❌ ERROR: Missing required column - {e}")
    print("   Please check your raw data file structure")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
