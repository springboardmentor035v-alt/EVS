import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

BASE_URL = "https://api.openaq.org/v3/locations"


page = 1
limit = 100
all_locations = []

headers = {"X-API-Key": API_KEY}

while True:
    params = {"country": "IN", "limit": limit, "page": page}
    resp = requests.get(BASE_URL, headers=headers, params=params)

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
        break

    all_locations.extend(results)
    print(f"Page {page} fetched, total locations so far: {len(all_locations)}")
    page += 1
    time.sleep(1)

# Save to CSV
df = pd.DataFrame(all_locations)
df.to_csv("data/global_locations_cleaned.csv", index=False)
print(f"Saved {len(df)} locations to data/global_locations_cleaned.csv")
