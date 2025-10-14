import pandas as pd
import json
import joblib
import folium
from folium.plugins import HeatMap, MarkerCluster # Import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
LOCATIONS_FILE = "data/specific_locations_cleaned.csv"
POLLUTION_FILE = "data/pollution_data.csv"
MODEL_FILE = 'pollution_source_model.joblib'
SCALER_FILE = 'data_scaler.joblib'
OUTPUT_MAP_FILE = "pollution_map.html"

# --- PART 1: DATA PREPARATION ---
print("--- Part 1: Loading and Preparing Data ---")
df_locations = pd.read_csv(LOCATIONS_FILE)
df_pollution = pd.read_csv(POLLUTION_FILE)
df = pd.merge(df_pollution, df_locations, left_on='location_id', right_on='id', how='left')

def extract_coords(coord_str):
    try:
        valid_json_str = str(coord_str).replace("'", '"')
        coord_dict = json.loads(valid_json_str)
        return coord_dict.get('latitude'), coord_dict.get('longitude')
    except (TypeError, json.JSONDecodeError): return None, None
df['latitude'], df['longitude'] = zip(*df['coordinates'].apply(extract_coords))
df.dropna(subset=['latitude', 'longitude'], inplace=True)
pollutant_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi']
for col in pollutant_cols:
    df[col] = df[col].fillna(df[col].median())

# --- PART 2: MODEL TRAINING ---
print("\n--- Part 2: Labeling Data and Training Model ---")
def label_pollution_source(row):
    if row['no2'] > 40: return 'Vehicular'
    if row['so2'] > 20: return 'Industrial'
    if row['pm10'] > 50:
        if (row['pm2_5'] / (row['pm10'] + 1e-6)) > 0.5: return 'Burning'
        else: return 'Dust'
    return 'Other'
df['pollution_source'] = df.apply(label_pollution_source, axis=1)

features_to_use = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi', 'latitude', 'longitude']
X = df[features_to_use]
y = df['pollution_source']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
print("Model and scaler have been trained and saved.")

# --- PART 3: GEOSPATIAL VISUALIZATION ---
print("\n--- Part 3: Generating Upgraded Interactive Map ---")
X_full_scaled = scaler.transform(X)
df['predicted_source'] = model.predict(X_full_scaled)
df['prediction_confidence'] = model.predict_proba(X_full_scaled).max(axis=1)
df.loc[df['prediction_confidence'] < 0.6, 'predicted_source'] = "Uncertain"

map_center = [df['latitude'].mean(), df['longitude'].mean()]
pollution_map = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

# ⭐ VISUALIZATION UPGRADE: Layered Heatmaps
heat_data_aqi = df[['latitude', 'longitude', 'aqi']].values.tolist()
HeatMap(heat_data_aqi, radius=15, name='Overall AQI Heatmap', show=True).add_to(pollution_map)
heat_data_no2 = df[['latitude', 'longitude', 'no2']].values.tolist()
HeatMap(heat_data_no2, radius=15, name='NO₂ (Traffic) Heatmap', show=False).add_to(pollution_map)

# ⭐ VISUALIZATION UPGRADE: Color-coded Source Clusters
marker_cluster = MarkerCluster(name="Source Clusters").add_to(pollution_map)
color_map = {'Vehicular':'blue', 'Industrial':'red', 'Burning':'orange', 'Dust':'beige', 'Other':'gray', 'Uncertain':'black'}
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color_map.get(row['predicted_source'], 'black'),
        fill=True,
        fill_opacity=0.7,
        popup=f"<b>{row['name']}</b><br>Source: {row['predicted_source']}<br>Confidence: {row['prediction_confidence']:.2%}"
    ).add_to(marker_cluster)

# Add layer control and save
folium.LayerControl().add_to(pollution_map)
pollution_map.save(OUTPUT_MAP_FILE)
print(f"\n✅ Upgraded map saved! Open '{OUTPUT_MAP_FILE}' to see the polished version.")