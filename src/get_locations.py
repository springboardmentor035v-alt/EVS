import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

# -------------------- CONFIGURATION --------------------
load_dotenv()
BASE_URL = "https://api.openaq.org/v3/locations"
API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError("❌ No API key found. Please set OPENAQ_API_KEY in your .env file.")

page = 1
limit = 100

# -------------------- ENHANCED MANUAL LOCATIONS --------------------
all_locations_data = [
    # ======= DELHI =======
    {'id': 'delhi-anand-vihar', 'name': 'Anand Vihar', 'city': 'Delhi', 'country': 'IN', 'coordinates': {'latitude': 28.646, 'longitude': 77.315}},
    {'id': 'delhi-rk-puram', 'name': 'R.K. Puram', 'city': 'Delhi', 'country': 'IN', 'coordinates': {'latitude': 28.562, 'longitude': 77.183}},
    {'id': 'delhi-mandir-marg', 'name': 'Mandir Marg', 'city': 'Delhi', 'country': 'IN', 'coordinates': {'latitude': 28.637, 'longitude': 77.207}},
    {'id': 'delhi-ashok-vihar', 'name': 'Ashok Vihar', 'city': 'Delhi', 'country': 'IN', 'coordinates': {'latitude': 28.692, 'longitude': 77.17}},

    # ======= BANGALORE =======
    {'id': 'bangalore-btm', 'name': 'BTM Layout', 'city': 'Bangalore', 'country': 'IN', 'coordinates': {'latitude': 12.916, 'longitude': 77.610}},
    {'id': 'bangalore-peenya', 'name': 'Peenya', 'city': 'Bangalore', 'country': 'IN', 'coordinates': {'latitude': 13.03, 'longitude': 77.53}},
    {'id': 'bangalore-yeshwanthpur', 'name': 'Yeshwanthpur', 'city': 'Bangalore', 'country': 'IN', 'coordinates': {'latitude': 13.02, 'longitude': 77.55}},
    {'id': 'bangalore-whitefield', 'name': 'Whitefield', 'city': 'Bangalore', 'country': 'IN', 'coordinates': {'latitude': 12.969, 'longitude': 77.749}},

    # ======= MUMBAI =======
    {'id': 'mumbai-bandra', 'name': 'Bandra', 'city': 'Mumbai', 'country': 'IN', 'coordinates': {'latitude': 19.054, 'longitude': 72.840}},
    {'id': 'mumbai-kurla', 'name': 'Kurla', 'city': 'Mumbai', 'country': 'IN', 'coordinates': {'latitude': 19.074, 'longitude': 72.880}},
    {'id': 'mumbai-borivali', 'name': 'Borivali', 'city': 'Mumbai', 'country': 'IN', 'coordinates': {'latitude': 19.231, 'longitude': 72.856}},
    {'id': 'mumbai-dadar', 'name': 'Dadar', 'city': 'Mumbai', 'country': 'IN', 'coordinates': {'latitude': 19.017, 'longitude': 72.843}},
]

headers = {"X-API-Key": API_KEY}
INDIA_BBOX = "68.1,8.0,97.4,37.1"

# -------------------- FETCH LOCATIONS FROM OPENAQ --------------------
while True:
    params = {"bbox": INDIA_BBOX, "limit": limit, "page": page, "entity": "government"}
    try:
        resp = requests.get(BASE_URL, headers=headers, params=params, timeout=20)
        if resp.status_code == 429:
            print("Rate limit hit. Waiting 10 seconds...")
            time.sleep(10)
            continue
        elif resp.status_code != 200:
            print(f"Error fetching data: {resp.status_code} - {resp.text}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            print("No more results to fetch.")
            break

        all_locations_data.extend(results)
        print(f"Page {page} fetched, total locations so far: {len(all_locations_data)}")
        page += 1
        time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        break

# -------------------- SAVE TO CSV --------------------
flat_locations = []
for loc in all_locations_data:
    flat_locations.append({
        'id': loc.get('id'),
        'name': loc.get('name'),
        'city': loc.get('city'),
        'country': loc.get('country'),
        'coordinates': loc.get('coordinates')
    })

df = pd.DataFrame(flat_locations)
df.drop_duplicates(subset='id', keep='first', inplace=True)
df.to_csv("data/global_locations_cleaned.csv", index=False)

print(f"\n✅ Saved {len(df)} unique locations (including subzones) to data/global_locations_cleaned.csv")
