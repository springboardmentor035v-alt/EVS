# scripts/01_data_collection.py
import pandas as pd
import requests
import osmnx as ox
import os
import config  # CORRECTED IMPORT
import logging

# The rest of the file is the same...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_feature_count(point, tags, dist=2000):
    try:
        gdf = ox.features_from_point(point, tags, dist=dist)
        return len(gdf)
    except Exception:
        return 0

def collect_all_data():
    logging.info("🚀 Starting Module 1: Data Collection")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    if not config.API_KEY:
        logging.error("❌ ERROR: OPENWEATHER_API_KEY not found. Check your .env file.")
        return

    try:
        locations_df = pd.read_csv(config.LOCATIONS_FILE)
    except FileNotFoundError:
        logging.error(f"❌ ERROR: Locations file not found at '{config.LOCATIONS_FILE}'")
        return

    all_air_data, all_weather_data, all_osm_features = [], [], []

    for _, row in locations_df.iterrows():
        city, lat, lon = row["name"], row["latitude"], row["longitude"]
        logging.info(f"--- Processing {city} ({lat}, {lon}) ---")

        # 1. Air Quality Data
        try:
            params = {"lat": lat, "lon": lon, "appid": config.API_KEY}
            response = requests.get("http://api.openweathermap.org/data/2.5/air_pollution", params=params, timeout=15)
            response.raise_for_status()
            air_data_list = response.json()["list"]
            if air_data_list:
                air_components = air_data_list[0]['components']
                air_components['name'] = city
                air_components['latitude'] = lat
                air_components['longitude'] = lon
                air_components['timestamp'] = air_data_list[0]['dt']
                all_air_data.append(air_components)
                logging.info(f"✅ Air quality fetched for {city}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Could not fetch air quality for {city}: {e}")

        # 2. Weather Data
        try:
            params = {"lat": lat, "lon": lon, "appid": config.API_KEY, "units": "metric"}
            response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=15)
            response.raise_for_status()
            weather = response.json()
            all_weather_data.append({
                "name": city, "latitude": lat, "longitude": lon,
                "temperature": weather["main"]["temp"], "humidity": weather["main"]["humidity"],
                "wind_speed": weather["wind"]["speed"], "timestamp": weather["dt"]
            })
            logging.info(f"✅ Weather fetched for {city}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Could not fetch weather for {city}: {e}")
            
        # 3. OSM Features
        point = (lat, lon)
        all_osm_features.append({
            "name": city, "latitude": lat, "longitude": lon,
            "roads_count": get_feature_count(point, tags={'highway': True}),
            "industrial_count": get_feature_count(point, tags={'landuse': 'industrial'}),
            "agriculture_count": get_feature_count(point, tags={'landuse': 'farmland'}),
            "dumps_count": get_feature_count(point, tags={'landuse': 'landfill'})
        })
        logging.info(f"✅ OSM features fetched for {city}")

    pd.DataFrame(all_air_data).to_csv(config.AIR_QUALITY_FILE, index=False)
    pd.DataFrame(all_weather_data).to_csv(config.WEATHER_FILE, index=False)
    pd.DataFrame(all_osm_features).to_csv(config.OSM_FEATURES_FILE, index=False)
    logging.info(f"\n✅ Module 1 complete. Raw data saved in '{config.OUTPUT_DIR}'.")

if __name__ == "__main__":
    collect_all_data()