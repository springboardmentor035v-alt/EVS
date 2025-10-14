import pandas as pd
import json
import joblib
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import folium
from folium.plugins import HeatMap
import sys

# --- CONFIGURATION ---
LOCATIONS_FILE = "data/global_locations_cleaned.csv"
POLLUTION_FILE = "data/pollution_data.csv"
MODEL_FILE = 'pollution_source_model.joblib'
SCALER_FILE = 'data_scaler.joblib'
OUTPUT_MAP_FILE = "pollution_map.html"

# ==============================================================================
# PART 1: DATA LOADING AND PREPARATION
# ==============================================================================
print("--- Part 1: Loading and Preparing Data ---")
df_locations = pd.read_csv(LOCATIONS_FILE)
df_pollution = pd.read_csv(POLLUTION_FILE)

# Convert both key columns to string to ensure a proper merge
df_pollution['location_id'] = df_pollution['location_id'].astype(str)
df_locations['id'] = df_locations['id'].astype(str)

df = pd.merge(df_pollution, df_locations, left_on='location_id', right_on='id', how='left')

# Safely extract and clean coordinates
def extract_coords(coord_str):
    try:
        valid_json_str = str(coord_str).replace("'", '"')
        coord_dict = json.loads(valid_json_str)
        return coord_dict.get('latitude'), coord_dict.get('longitude')
    except (TypeError, json.JSONDecodeError):
        return None, None

df['latitude'], df['longitude'] = zip(*df['coordinates'].apply(extract_coords))
df.dropna(subset=['latitude', 'longitude'], inplace=True)

# --- FIX: ADD CHECK FOR EMPTY DATAFRAME ---
if df.empty:
    print("\nERROR: DataFrame is empty after loading and merging source files.")
    print("This likely means no matching locations were found between your pollution and locations CSV files.")
    print("Please check that 'data/pollution_data.csv' is not empty and re-run the data gathering scripts if needed.")
    sys.exit() # Stop the script gracefully
# ----------------------------------------------

pollutant_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi']
for col in pollutant_cols:
    df[col] = df[col].fillna(df[col].median())

print("Data loaded successfully.")

# ==============================================================================
# PART 2: GEOSPATIAL FEATURE ENGINEERING
# ==============================================================================
print("\n--- Part 2: Engineering Geospatial Features with OSMnx ---")
print("This may take a while as it downloads map data...")

def get_distance_to_nearest(lat, lon, tags):
    """Calculates distance in meters to the nearest feature."""
    try:
        gdf = ox.geometries_from_point((lat, lon), tags, dist=1500) # 1.5km radius
        if gdf.empty:
            return None
        point_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        point_proj = point_geom.to_crs(gdf.crs)
        return gdf.distance(point_proj.iloc[0]).min()
    except Exception:
        return None

df['dist_to_road_m'] = df.apply(
    lambda row: get_distance_to_nearest(row['latitude'], row['longitude'], {'highway': ['primary', 'secondary', 'motorway']}),
    axis=1
)
df['dist_to_industrial_m'] = df.apply(
    lambda row: get_distance_to_nearest(row['latitude'], row['longitude'], {'landuse': 'industrial'}),
    axis=1
)

# Fix for FutureWarning: Use direct assignment instead of inplace on a slice
df['dist_to_road_m'] = df['dist_to_road_m'].fillna(1500)
df['dist_to_industrial_m'] = df['dist_to_industrial_m'].fillna(1500)

print("Geospatial features created and added to the dataset.")

# ==============================================================================
# PART 3: SOURCE LABELING AND MODEL TRAINING
# ==============================================================================
print("\n--- Part 3: Labeling Data and Training Upgraded Model ---")

def label_pollution_source(row):
    if row['no2'] > 35 and row['dist_to_road_m'] < 200:
        return 'Vehicular'
    if row['so2'] > 15 and row['dist_to_industrial_m'] < 500:
        return 'Industrial'
    if row['pm10'] > 50:
        if (row['pm2_5'] / (row['pm10'] + 1e-6)) > 0.5:
            return 'Burning'
        else:
            return 'Dust'
    return 'Other'

df['pollution_source'] = df.apply(label_pollution_source, axis=1)

features_to_use = [
    'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi',
    'latitude', 'longitude', 'dist_to_road_m', 'dist_to_industrial_m'
]
X = df[features_to_use]
y = df['pollution_source']

if y.nunique() > 1 and all(y.value_counts() >= 2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print("Model and scaler have been trained and saved.")

print("\n--- Model Performance Report ---")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# ==============================================================================
# PART 4: GEOSPATIAL VISUALIZATION & DATA EXPORT
# ==============================================================================
print("\n--- Part 4: Generating New Interactive Map ---")

X_full_scaled = scaler.transform(X)
df['predicted_source'] = model.predict(X_full_scaled)

map_center = [df['latitude'].mean(), df['longitude'].mean()]
pollution_map = folium.Map(location=map_center, zoom_start=11)

source_styles = {
    'Vehicular': {'color': 'blue', 'icon': 'car'},
    'Industrial': {'color': 'red', 'icon': 'industry'},
    'Burning': {'color': 'orange', 'icon': 'fire'},
    'Dust': {'color': 'beige', 'icon': 'cloud'},
    'Other': {'color': 'gray', 'icon': 'question-sign'}
}

source_markers = folium.FeatureGroup(name='Pollution Sources')
for idx, row in df.iterrows():
    source = row['predicted_source']
    style = source_styles.get(source, source_styles['Other'])
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"<strong>Source: {source}</strong><br>PM2.5: {row['pm2_5']:.2f}<br>Dist to Road: {row['dist_to_road_m']:.0f}m",
        icon=folium.Icon(color=style['color'], icon=style['icon'], prefix='fa')
    ).add_to(source_markers)
source_markers.add_to(pollution_map)

heat_data = df[['latitude', 'longitude', 'aqi']].values.tolist()
HeatMap(heat_data, radius=15).add_to(folium.FeatureGroup(name='AQI Heatmap').add_to(pollution_map))

folium.LayerControl().add_to(pollution_map)
pollution_map.save(OUTPUT_MAP_FILE)

# Save the final data for the dashboard to use
df.to_csv("data/final_predictions.csv", index=False)

print(f"\n✅ Workflow complete! Open '{OUTPUT_MAP_FILE}' to view your new map.")
print("The final data has been saved to 'data/final_predictions.csv' for the dashboard.")