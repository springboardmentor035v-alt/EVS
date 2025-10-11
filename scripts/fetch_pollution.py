# scripts/fetch_pollution.py
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")
LOCATIONS_FILE = "data/global_locations_cleaned.csv"
OUTPUT_FILE = "data/pollution_data.csv"
API_KEY = "e0f3cf0d7c2c77cb00a6e0d258cee192"  # your OpenWeather API key
POLLUTANTS = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co']
SLEEP_TIME = 1  # seconds to avoid rate limits

# -------------------- LOAD LOCATIONS --------------------
df_locations = pd.read_csv(LOCATIONS_FILE)

# Extract latitude/longitude from coordinates column if needed
def extract_lat_lon(coord_str):
    try:
        coord_dict = eval(coord_str)
        return float(coord_dict['latitude']), float(coord_dict['longitude'])
    except:
        return None, None

df_locations['latitude'], df_locations['longitude'] = zip(*df_locations['coordinates'].map(extract_lat_lon))

# -------------------- FUNCTION TO FETCH DATA --------------------
def fetch_latest_pollution(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'list' in data and len(data['list']) > 0:
                components = data['list'][0].get('components', {})
                aqi = data['list'][0].get('main', {}).get('aqi')
                measures = {p: components.get(p) for p in POLLUTANTS}
                measures['aqi'] = aqi
                return measures
        return {p: None for p in POLLUTANTS + ['aqi']}
    except Exception as e:
        print(f"Error fetching data for {lat}, {lon}: {e}")
        return {p: None for p in POLLUTANTS + ['aqi']}

# -------------------- FETCH DATA FOR ALL LOCATIONS --------------------
results = []  # collect data in a list to avoid FutureWarning
counter = 0   # counter for processed locations

for idx, row in df_locations.iterrows():
    loc_id = row.get("id")
    lat, lon = row.get("latitude"), row.get("longitude")
    if pd.isnull(lat) or pd.isnull(lon):
        continue

    measures = fetch_latest_pollution(lat, lon)
    data = {'location_id': loc_id}
    data.update(measures)
    results.append(data)  # append to list
    counter += 1

    # print every 50 processed locations
    if counter % 50 == 0:
        print(f"{counter} locations processed...")

    time.sleep(SLEEP_TIME)

# -------------------- CONVERT TO DATAFRAME & SAVE --------------------
pollution_df = pd.DataFrame(results)
pollution_df.to_csv(OUTPUT_FILE, index=False)
print(f"Realtime pollution data saved to {OUTPUT_FILE}")
print(f"Total locations processed: {counter}")
