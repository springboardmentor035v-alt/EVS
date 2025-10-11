# scripts/fetch_weather.py

import pandas as pd
import requests
import time
import os
import ast
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OWM_API_KEY")
LOCATIONS_FILE = "data/global_locations_cleaned.csv"
WEATHER_FILE = "data/weather_data.csv"

# Load locations
df_locations = pd.read_csv(LOCATIONS_FILE)

# --- Parse latitude and longitude from 'coordinates' column ---
def parse_coords(coord_str):
    if pd.isnull(coord_str) or coord_str == '':
        return None, None
    try:
        coord = ast.literal_eval(coord_str)
        return coord.get('latitude'), coord.get('longitude')
    except:
        return None, None

df_locations['latitude'], df_locations['longitude'] = zip(*df_locations['coordinates'].apply(parse_coords))

# --- Initialize or load existing weather data ---
if os.path.exists(WEATHER_FILE):
    try:
        df_weather = pd.read_csv(WEATHER_FILE)
        if df_weather.empty:
            df_weather = pd.DataFrame(columns=['location_id', 'temperature', 'humidity', 'wind_speed'])
            fetched_ids = set()
        else:
            fetched_ids = set(df_weather['location_id'])
    except pd.errors.EmptyDataError:
        df_weather = pd.DataFrame(columns=['location_id', 'temperature', 'humidity', 'wind_speed'])
        fetched_ids = set()
else:
    df_weather = pd.DataFrame(columns=['location_id', 'temperature', 'humidity', 'wind_speed'])
    fetched_ids = set()

total_locations = len(df_locations)
print(f"Total locations to fetch weather for: {total_locations}")

# --- Fetch weather data ---
for idx, row in df_locations.iterrows():
    loc_id = row['id']
    if loc_id in fetched_ids:
        continue  # skip already fetched

    lat, lon = row['latitude'], row['longitude']
    if pd.isnull(lat) or pd.isnull(lon):
        continue

    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            weather = resp.json()
            df_weather = pd.concat([df_weather, pd.DataFrame([{
                "location_id": loc_id,
                "temperature": weather['main']['temp'],
                "humidity": weather['main']['humidity'],
                "wind_speed": weather['wind']['speed']
            }])], ignore_index=True)
        else:
            print(f"Failed for ID {loc_id}: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Exception for ID {loc_id}: {e}")

    # Save progress every 50 locations
    if idx % 50 == 0:
        df_weather.to_csv(WEATHER_FILE, index=False)
        print(f"Saved {len(df_weather)} weather entries so far...")

    time.sleep(1)  # avoid hitting rate limit

# --- Final save ---
df_weather.to_csv(WEATHER_FILE, index=False)
print(f"Weather data fetch complete. Total records saved: {len(df_weather)}")
