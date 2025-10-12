import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
LOCATIONS_FILE = "data/global_locations_cleaned.csv"
OUTPUT_FILE = "data/pollution_data.csv"
API_KEY = os.getenv("OPENWEATHER_API_KEY")
POLLUTANTS = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co']
SLEEP_TIME = 0.5  # Can be slightly faster as retries will handle rate limits

# --- SETUP ROBUST REQUESTS SESSION ---
def create_requests_session():
    """Creates a requests session with a retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1  # Wait 1s, 2s, 4s, etc. between retries
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# --- MAIN SCRIPT LOGIC ---
if not API_KEY:
    raise ValueError("No API key found. Please set OPENWEATHER_API_KEY in your .env file.")

# Load locations
df_locations = pd.read_csv(LOCATIONS_FILE)

# --- CHECKPOINTING: Find which locations are already processed ---
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    print("Output file found. Resuming from last checkpoint...")
    df_processed = pd.read_csv(OUTPUT_FILE)
    processed_ids = set(df_processed['location_id'].astype(str))
    print(f"{len(processed_ids)} locations already processed. Skipping them.")

# Filter out locations that are already processed
df_locations['id'] = df_locations['id'].astype(str)
df_to_process = df_locations[~df_locations['id'].isin(processed_ids)]

print(f"Total locations to process: {len(df_to_process)} out of {len(df_locations)}")

# Function to fetch data using the robust session
def fetch_latest_pollution(session, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        # Use a longer timeout to be safe
        resp = session.get(url, timeout=20)
        resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = resp.json()
        if 'list' in data and len(data['list']) > 0:
            components = data['list'][0].get('components', {})
            aqi = data['list'][0].get('main', {}).get('aqi')
            measures = {p: components.get(p) for p in POLLUTANTS}
            measures['aqi'] = aqi
            return measures
        return {p: None for p in POLLUTANTS + ['aqi']}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {lat}, {lon}: {e}")
        return {p: None for p in POLLUTANTS + ['aqi']}

# Create the session and start fetching
session = create_requests_session()
counter = 0

# Open file in append mode to save progress
with open(OUTPUT_FILE, 'a', newline='') as f:
    # Write header only if the file is new/empty
    if not processed_ids:
        pd.DataFrame(columns=['location_id'] + POLLUTANTS + ['aqi']).to_csv(f, index=False, header=True)

    for idx, row in df_to_process.iterrows():
        loc_id = row.get("id")
        # Ensure coordinates are extracted correctly
        try:
            coord_dict = eval(row['coordinates'])
            lat, lon = float(coord_dict['latitude']), float(coord_dict['longitude'])
        except (TypeError, ValueError, SyntaxError):
            print(f"Skipping location {loc_id} due to invalid coordinates.")
            continue

        if pd.isnull(lat) or pd.isnull(lon):
            continue

        measures = fetch_latest_pollution(session, lat, lon)
        data_row = {'location_id': loc_id}
        data_row.update(measures)
        
        # Save each row immediately
        pd.DataFrame([data_row]).to_csv(f, index=False, header=False)
        counter += 1

        if counter % 50 == 0:
            print(f"{counter} new locations processed...")

        time.sleep(SLEEP_TIME)

print("\n✅ Data enrichment complete!")
print(f"Total new locations processed in this run: {counter}")