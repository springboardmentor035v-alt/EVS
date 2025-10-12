import os
import requests
import pandas as pd
import numpy as np
import osmnx as ox
from dotenv import load_dotenv
from openaq_utils.openaq_sensors import get_sensors
from weather_utils.current_weather import get_current_weather

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# --- Cities with bounding boxes (more cities for more data) ---
cities = {
    "Delhi": {"min_lat": 28.4, "max_lat": 28.9, "min_lon": 76.8, "max_lon": 77.4},
    "Mumbai": {"min_lat": 18.9, "max_lat": 19.3, "min_lon": 72.7, "max_lon": 72.95},
    "Bangalore": {"min_lat": 12.85, "max_lat": 13.1, "min_lon": 77.5, "max_lon": 77.7},
    "Chennai": {"min_lat": 13.0, "max_lat": 13.2, "min_lon": 80.2, "max_lon": 80.35},
    "Kolkata": {"min_lat": 22.5, "max_lat": 22.7, "min_lon": 88.3, "max_lon": 88.45},
    "Hyderabad": {"min_lat": 17.35, "max_lat": 17.5, "min_lon": 78.35, "max_lon": 78.55},
    "Pune": {"min_lat": 18.45, "max_lat": 18.65, "min_lon": 73.75, "max_lon": 73.95},
    "Ahmedabad": {"min_lat": 23.0, "max_lat": 23.1, "min_lon": 72.55, "max_lon": 72.65},
    "Jaipur": {"min_lat": 26.85, "max_lat": 27.0, "min_lon": 75.75, "max_lon": 75.9},
    "Lucknow": {"min_lat": 26.8, "max_lat": 26.95, "min_lon": 80.9, "max_lon": 81.0},
    "Kanpur": {"min_lat": 26.4, "max_lat": 26.5, "min_lon": 80.3, "max_lon": 80.4},
    "Nagpur": {"min_lat": 21.1, "max_lat": 21.2, "min_lon": 79.0, "max_lon": 79.15},
    "Indore": {"min_lat": 22.7, "max_lat": 22.85, "min_lon": 75.8, "max_lon": 75.95},
    "Thane": {"min_lat": 19.15, "max_lat": 19.25, "min_lon": 72.95, "max_lon": 73.05},
    "Bhopal": {"min_lat": 23.2, "max_lat": 23.35, "min_lon": 77.35, "max_lon": 77.5},
    "Visakhapatnam": {"min_lat": 17.65, "max_lat": 17.8, "min_lon": 83.2, "max_lon": 83.35},
    "Pimpri-Chinchwad": {"min_lat": 18.55, "max_lat": 18.65, "min_lon": 73.75, "max_lon": 73.85},
    "Patna": {"min_lat": 25.55, "max_lat": 25.7, "min_lon": 85.05, "max_lon": 85.15},
    "Vadodara": {"min_lat": 22.3, "max_lat": 22.4, "min_lon": 73.15, "max_lon": 73.25},
    "Ghaziabad": {"min_lat": 28.65, "max_lat": 28.75, "min_lon": 77.4, "max_lon": 77.5},
    "Ludhiana": {"min_lat": 30.9, "max_lat": 31.05, "min_lon": 75.75, "max_lon": 75.9},
    "Agra": {"min_lat": 27.15, "max_lat": 27.25, "min_lon": 77.95, "max_lon": 78.05},
    "Nashik": {"min_lat": 19.95, "max_lat": 20.05, "min_lon": 73.75, "max_lon": 73.9},
    "Faridabad": {"min_lat": 28.35, "max_lat": 28.45, "min_lon": 77.3, "max_lon": 77.45},
    "Meerut": {"min_lat": 28.95, "max_lat": 29.05, "min_lon": 77.7, "max_lon": 77.85},
    "Rajkot": {"min_lat": 22.25, "max_lat": 22.35, "min_lon": 70.75, "max_lon": 70.9},
    "Kalyan": {"min_lat": 19.2, "max_lat": 19.3, "min_lon": 73.15, "max_lon": 73.25},
    "Vasai-Virar": {"min_lat": 19.3, "max_lat": 19.45, "min_lon": 72.8, "max_lon": 73.0},
    "Varanasi": {"min_lat": 25.25, "max_lat": 25.35, "min_lon": 82.95, "max_lon": 83.05},
    "Srinagar": {"min_lat": 34.05, "max_lat": 34.15, "min_lon": 74.75, "max_lon": 74.9},
    "Aurangabad": {"min_lat": 19.85, "max_lat": 19.95, "min_lon": 75.3, "max_lon": 75.45},
    "Dhanbad": {"min_lat": 23.75, "max_lat": 23.85, "min_lon": 86.4, "max_lon": 86.55},
    "Amritsar": {"min_lat": 31.6, "max_lat": 31.7, "min_lon": 74.8, "max_lon": 74.95},
    "Allahabad": {"min_lat": 25.4, "max_lat": 25.55, "min_lon": 81.8, "max_lon": 81.95},
    "Ranchi": {"min_lat": 23.33, "max_lat": 23.45, "min_lon": 85.28, "max_lon": 85.42},
    "Howrah": {"min_lat": 22.57, "max_lat": 22.65, "min_lon": 88.3, "max_lon": 88.35},
    "Coimbatore": {"min_lat": 11.0, "max_lat": 11.1, "min_lon": 76.9, "max_lon": 77.0},
    "Jabalpur": {"min_lat": 23.15, "max_lat": 23.25, "min_lon": 79.9, "max_lon": 80.0},
    "Guwahati": {"min_lat": 26.15, "max_lat": 26.25, "min_lon": 91.7, "max_lon": 91.85},
    "Jodhpur": {"min_lat": 26.25, "max_lat": 26.35, "min_lon": 73.0, "max_lon": 73.15},
    "Madurai": {"min_lat": 9.9, "max_lat": 10.05, "min_lon": 78.1, "max_lon": 78.25},
    "Tiruchirappalli": {"min_lat": 10.75, "max_lat": 10.85, "min_lon": 78.65, "max_lon": 78.8},
    "Bhubaneswar": {"min_lat": 20.25, "max_lat": 20.35, "min_lon": 85.8, "max_lon": 85.95},
    "Salem": {"min_lat": 11.6, "max_lat": 11.7, "min_lon": 78.1, "max_lon": 78.25},
    "Mangalore": {"min_lat": 12.85, "max_lat": 12.95, "min_lon": 74.85, "max_lon": 75.0},
    "Jamshedpur": {"min_lat": 22.75, "max_lat": 22.85, "min_lon": 86.15, "max_lon": 86.3},
    "Ujjain": {"min_lat": 23.15, "max_lat": 23.25, "min_lon": 75.75, "max_lon": 75.9},
    "Tiruppur": {"min_lat": 11.1, "max_lat": 11.2, "min_lon": 77.3, "max_lon": 77.45},
    "Gorakhpur": {"min_lat": 26.7, "max_lat": 26.8, "min_lon": 83.35, "max_lon": 83.5},
    "Jalandhar": {"min_lat": 31.3, "max_lat": 31.45, "min_lon": 75.55, "max_lon": 75.7},
    "Belgaum": {"min_lat": 15.85, "max_lat": 15.95, "min_lon": 74.5, "max_lon": 74.65},
}

# --- Function to fetch OpenAQ + Weather + OSM features ---
def get_values(lat, lon, required_params=None, radius=5000, osm_dist=2000):
    if required_params is None:
        required_params = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']

    # OpenAQ sensors
    stations = get_sensors(lat, lon, radius=radius, limit=10)
    if not stations:
        row = {p: None for p in required_params}
        row.update({
            'latitude': lat,
            'longitude': lon,
            'station_id': None,
            'station_name': None,
            'weather': get_current_weather(lat, lon, WEATHER_KEY)
        })
    else:
        nearest_id, nearest = min(
            stations.items(),
            key=lambda kv: kv[1].get('distance_m') or float('inf')
        )
        row = {p: None for p in required_params}
        row.update({
            'latitude': lat,
            'longitude': lon,
            'station_id': nearest_id,
            'station_name': nearest.get('station_name'),
            'weather': get_current_weather(lat, lon, WEATHER_KEY)
        })
        for s in nearest.get('sensors', []):
            param = s.get('parameter')
            if param in required_params:
                sid = s.get('sensor_id')
                meas_url = f"https://api.openaq.org/v3/sensors/{sid}/measurements"
                r = requests.get(meas_url, headers=HEADERS, params={"limit":1, "sort":"desc"})
                if r.status_code == 200:
                    mvals = r.json().get('results', [])
                    if mvals:
                        row[param] = mvals[0].get('value')

    # OSM features
    tags = {"landuse": ["industrial", "farmland", "farmyard"], "amenity": ["waste_disposal", "recycling"]}
    try:
        landuse_gdf = ox.features_from_point((lat, lon), dist=osm_dist, tags=tags)
        row['num_industrial'] = len(landuse_gdf[landuse_gdf.get('landuse') == 'industrial']) if 'landuse' in landuse_gdf.columns else 0
        row['num_farmland'] = len(landuse_gdf[landuse_gdf.get('landuse') == 'farmland']) if 'landuse' in landuse_gdf.columns else 0
        row['num_dumpsites'] = len(landuse_gdf[landuse_gdf.get('amenity') == 'waste_disposal']) if 'amenity' in landuse_gdf.columns else 0
        row['num_recycling'] = len(landuse_gdf[landuse_gdf.get('amenity') == 'recycling']) if 'amenity' in landuse_gdf.columns else 0
    except:
        row.update({'num_industrial': 0, 'num_farmland':0, 'num_dumpsites':0, 'num_recycling':0})

    return row


locations = []
step = 0.10  # ~2 km spacing for more points
for city, bbox in cities.items():
    lats = np.arange(bbox["min_lat"], bbox["max_lat"], step)
    lons = np.arange(bbox["min_lon"], bbox["max_lon"], step)
    locations += [(lat, lon) for lat in lats for lon in lons]

print(f"Total sampling points: {len(locations)}")

# --- Fetch data ---
rows = []
for i, (lat, lon) in enumerate(locations):
    print(f"Fetching point {i+1}/{len(locations)}: ({lat},{lon})")
    try:
        data = get_values(lat, lon)
        rows.append(data)
    except Exception as e:
        print(f"Error at ({lat},{lon}): {e}")

# --- Convert to DataFrame ---
df = pd.DataFrame(rows)
df['source_air_quality'] = 'OpenAQ'
df['source_weather'] = 'OpenWeatherMap'
df['timestamp'] = pd.Timestamp.now()

# --- Save ---
os.makedirs("./data", exist_ok=True)
df.to_csv("./data/module1_data_training.csv", index=False)
df.to_json("./data/module1_data_training.json", orient='records', lines=True)

print(f"Total records collected: {len(df)}")
print(df.head())
