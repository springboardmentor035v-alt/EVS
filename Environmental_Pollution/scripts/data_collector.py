# scripts/data_collector.py

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import time
from datetime import datetime, timedelta
import calendar
import os
import concurrent.futures

# Import configuration and utility functions
import config
from utils import save_to_csv, save_to_json

def convert_date_to_unix(date_str):
    """Converts a YYYY-MM-DD string to a Unix timestamp."""
    return int(time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple()))

def create_retry_session(max_retries=3):
    """Creates a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=2,  # Wait 2, 4, 8 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_single_request(city_name, lat, lon, start_date, end_date, session):
    """Attempts to fetch all data in a single request."""
    chunk_start = int(start_date.timestamp())
    chunk_end = int(end_date.timestamp())
    
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat,
        "lon": lon,
        "start": chunk_start,
        "end": chunk_end,
        "appid": config.OPENWEATHER_API_KEY
    }
    
    all_pollutant_records = []
    
    try:
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json().get('list', [])
        
        for hourly_data in data:
            timestamp = datetime.utcfromtimestamp(hourly_data['dt']).isoformat() + 'Z'
            components = hourly_data.get('components', {})
            
            pollutant_mapping = {
                "pm2_5": "pm25", "pm10": "pm10", "no2": "no2",
                "co": "co", "so2": "so2", "o3": "o3"
            }
            
            for key, value in components.items():
                if key in pollutant_mapping:
                    all_pollutant_records.append({
                        "location_name": city_name,
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": timestamp,
                        "pollutant": pollutant_mapping[key],
                        "value": value,
                        "unit": "µg/m³"
                    })
        
        print(f"  ✅ Completed AQ fetch for {city_name}: {len(data)} hourly records")
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Single request failed for {city_name}: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_pollutant_records)
    if not df.empty:
        df['aq_source_api'] = 'OpenWeatherMap'
    return df

def fetch_air_quality_data_owm(city_info, use_chunks=False):
    """
    Fetches historical air quality data from the OpenWeatherMap Air Pollution API.
    Fetches data in monthly chunks with retry logic to handle API failures.
    Set use_chunks=True if API times out, False for faster single request.
    """
    city_name, lat, lon, _ = city_info
    print(f"▶️ Starting OWM air quality fetch for {city_name}...")
    
    all_pollutant_records = []
    session = create_retry_session(max_retries=3)
    
    try:
        start_date = datetime(2020, 11, 28)
        end_date = datetime.now()
        
        # If not using chunks, try single request (much faster if API can handle it)
        if not use_chunks:
            return fetch_single_request(city_name, lat, lon, start_date, end_date, session)
        
        current = start_date
        successful_chunks = 0
        failed_chunks = 0
        
        while current < end_date:
            # Get month boundaries
            chunk_start = int(current.timestamp())
            next_month = current + timedelta(days=32)
            next_month = next_month.replace(day=1)
            chunk_end = min(int(next_month.timestamp()), int(end_date.timestamp()))
            
            url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
            params = {
                "lat": lat,
                "lon": lon,
                "start": chunk_start,
                "end": chunk_end,
                "appid": config.OPENWEATHER_API_KEY
            }
            
            try:
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json().get('list', [])
                
                for hourly_data in data:
                    timestamp = datetime.utcfromtimestamp(hourly_data['dt']).isoformat() + 'Z'
                    components = hourly_data.get('components', {})
                    
                    pollutant_mapping = {
                        "pm2_5": "pm25", "pm10": "pm10", "no2": "no2",
                        "co": "co", "so2": "so2", "o3": "o3"
                    }
                    
                    for key, value in components.items():
                        if key in pollutant_mapping:
                            all_pollutant_records.append({
                                "location_name": city_name,
                                "latitude": lat,
                                "longitude": lon,
                                "timestamp": timestamp,
                                "pollutant": pollutant_mapping[key],
                                "value": value,
                                "unit": "µg/m³"
                            })
                
                successful_chunks += 1
                # Reduced delay for faster processing
                time.sleep(0.1)  # Rate limiting between chunks
                
            except requests.exceptions.RequestException as e:
                failed_chunks += 1
                print(f"  ⚠️ Error for {city_name} chunk {current.strftime('%Y-%m')}: {e}")
            
            current = next_month
        
        if successful_chunks > 0:
            print(f"  ✅ Completed AQ fetch for {city_name}: {successful_chunks} successful, {failed_chunks} failed chunks")
        else:
            print(f"  ❌ Failed to fetch any data for {city_name}")
        
    except Exception as e:
        print(f"  ❌ Unexpected error fetching AQ data for {city_name}: {e}")
        return pd.DataFrame()
    finally:
        session.close()

    df = pd.DataFrame(all_pollutant_records)
    if not df.empty:
        df['aq_source_api'] = 'OpenWeatherMap'
    return df

def _fetch_single_weather_point(weather_request_info):
    """Helper function to fetch a single weather data point for concurrency."""
    lat, lon, date_obj = weather_request_info
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": config.OPENWEATHER_API_KEY, "units": "metric"}
    
    session = create_retry_session(max_retries=2)
    
    try:
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return {
            "latitude": lat, 
            "longitude": lon, 
            "date": date_obj,
            "temperature": data['main']['temp'], 
            "humidity": data['main']['humidity'],
            "wind_speed": data['wind']['speed'], 
            "wind_direction": data['wind'].get('deg', None)
        }
    except requests.exceptions.RequestException:
        return None
    finally:
        session.close()

def fetch_weather_data_concurrently(df):
    """Fetches weather data concurrently for all unique location-date pairs."""
    print("\nFetching weather data concurrently...")
    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    unique_points = df.drop_duplicates(subset=['latitude', 'longitude', 'date'])
    
    weather_requests = [(row['latitude'], row['longitude'], row['date']) for _, row in unique_points.iterrows()]
    
    weather_records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(_fetch_single_weather_point, weather_requests)
        
        for result in results:
            if result:
                weather_records.append(result)

    weather_df = pd.DataFrame(weather_records)
    if not weather_df.empty:
        weather_df['weather_source_api'] = 'OpenWeatherMap'
    print(f"  Collected {len(weather_df)} unique daily weather records.")
    return weather_df

def extract_geospatial_features(city_name, lat, lon, radius, tags):
    """Extracts distances to nearest OSM features."""
    print(f"\nExtracting geospatial features for {city_name}...")
    point = Point(lon, lat)
    features = {'latitude': lat, 'longitude': lon, 'geo_source_api': 'OpenStreetMap'}

    for feature_name, tag in tags.items():
        try:
            gdf = ox.features_from_point((lat, lon), tags=tag, dist=radius)
            if not gdf.empty:
                gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
                point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(gdf_proj.crs).iloc[0]
                distance = gdf_proj.distance(point_proj).min()
                features[f"distance_to_nearest_{feature_name}_m"] = distance
            else:
                features[f"distance_to_nearest_{feature_name}_m"] = None
        except Exception as e:
            print(f"  Could not extract '{feature_name}' for {city_name}: {e}")
            features[f"distance_to_nearest_{feature_name}_m"] = None
    return features

def run_pipeline():
    """Main function to run the entire data collection pipeline concurrently."""
    cities_df = pd.read_csv(config.CITIES_FILE)
    city_tuples = [tuple(x) for x in cities_df.to_numpy()]
    
    all_aq_dfs = []
    all_geo_features = []
    
    # Increased workers for faster parallel processing
    max_workers = min(10, len(city_tuples))  # Changed from 5 to 10
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"--- Starting concurrent fetch for {len(city_tuples)} cities (max {max_workers} workers) ---")
        
        future_to_city_aq = {executor.submit(fetch_air_quality_data_owm, city): city for city in city_tuples}
        future_to_city_geo = {executor.submit(extract_geospatial_features, city[0], city[1], city[2], config.GEOSPATIAL_RADIUS_METERS, config.OSM_TAGS): city for city in city_tuples}

        print("\nWaiting for Air Quality data...")
        for future in concurrent.futures.as_completed(future_to_city_aq):
            city_name = future_to_city_aq[future][0]
            try:
                df = future.result()
                if not df.empty:
                    all_aq_dfs.append(df)
            except Exception as e:
                print(f"  ❌ AQ fetch for {city_name} generated an exception: {e}")

        print("\nWaiting for Geospatial data...")
        for future in concurrent.futures.as_completed(future_to_city_geo):
            city_name = future_to_city_geo[future][0]
            try:
                features = future.result()
                all_geo_features.append(features)
                print(f"  ✅ Completed Geo fetch for {city_name}")
            except Exception as e:
                print(f"  ❌ Geo fetch for {city_name} generated an exception: {e}")

    if not all_aq_dfs:
        print("\n❌ No air quality data was collected. This may be due to API issues.")
        print("   Suggestions:")
        print("   1. Check if OpenWeatherMap API is operational")
        print("   2. Verify your API key is valid and has not exceeded rate limits")
        print("   3. Try running the script again in 30-60 minutes")
        print("   4. Consider reducing the date range or number of cities")
        return

    print("\n--- Consolidating all data ---")
    aq_dataset = pd.concat(all_aq_dfs, ignore_index=True)
    geo_dataset = pd.DataFrame(all_geo_features)

    weather_dataset = fetch_weather_data_concurrently(aq_dataset)

    print("\nMerging all datasets...")
    final_dataset = pd.merge(aq_dataset, geo_dataset, on=['latitude', 'longitude'], how='left')
    if not weather_dataset.empty:
        final_dataset['date'] = pd.to_datetime(final_dataset['timestamp']).dt.date
        final_dataset = pd.merge(final_dataset, weather_dataset, on=['latitude', 'longitude', 'date'], how='left')

    final_dataset = final_dataset.drop(columns=[col for col in ['date'] if col in final_dataset.columns])
    
    print(f"\n✅ Total records collected: {len(final_dataset)}")
    print("Final DataFrame columns:", final_dataset.columns.tolist())
    
    save_to_csv(final_dataset, config.DATA_DIR, "consolidated_enviro_data.csv")
    save_to_json(final_dataset, config.DATA_DIR, "consolidated_enviro_data.json")
    
    print("\n🎉 Data collection completed successfully!")

if __name__ == "__main__":
    run_pipeline()