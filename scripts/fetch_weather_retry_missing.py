# scripts/fetch_weather_retry_missing_fixed.py
import os
import time
import requests
import pandas as pd
import csv
import ast
import shutil
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")


LOCATIONS_FILE = "data/global_locations_cleaned.csv"
WEATHER_FILE = "data/weather_data.csv"
BACKUP_FILE = WEATHER_FILE + ".bak"

# 0) safety: ensure locations file exists
if not os.path.exists(LOCATIONS_FILE):
    raise SystemExit(f"Error: {LOCATIONS_FILE} not found")

# 1) backup existing weather file (if present)
if os.path.exists(WEATHER_FILE):
    shutil.copy2(WEATHER_FILE, BACKUP_FILE)
    print(f"Backup created: {BACKUP_FILE}")

# 2) load locations
df_loc = pd.read_csv(LOCATIONS_FILE)

# If 'latitude'/'longitude' aren't separate columns, try parsing coordinates column
if 'latitude' not in df_loc.columns or 'longitude' not in df_loc.columns:
    def parse_coords(s):
        if pd.isna(s) or s == "":
            return (None, None)
        try:
            d = ast.literal_eval(s)
            return d.get('latitude'), d.get('longitude')
        except Exception:
            return (None, None)
    df_loc['latitude'], df_loc['longitude'] = zip(*df_loc['coordinates'].apply(parse_coords))

# 3) read existing weather file safely
try:
    df_w = pd.read_csv(WEATHER_FILE)
except (pd.errors.EmptyDataError, FileNotFoundError):
    df_w = pd.DataFrame(columns=['location_id', 'temperature', 'humidity', 'wind_speed'])

# ensure numeric types for id columns
df_loc['id'] = pd.to_numeric(df_loc['id'], errors='coerce').astype('Int64')
if not df_w.empty:
    df_w['location_id'] = pd.to_numeric(df_w['location_id'], errors='coerce').astype('Int64')

loc_ids = set(df_loc['id'].dropna().astype(int))
have_ids = set(df_w['location_id'].dropna().astype(int)) if not df_w.empty else set()
missing_ids = sorted(loc_ids - have_ids)

print(f"Total locations: {len(loc_ids)}  |  Have weather: {len(have_ids)}  |  Missing: {len(missing_ids)}")

if not missing_ids:
    print("No missing ids. Nothing to do.")
    raise SystemExit(0)

# 4) prepare append CSV writer (write header only if file missing/empty)
write_header = not os.path.exists(WEATHER_FILE) or os.path.getsize(WEATHER_FILE) == 0
f = open(WEATHER_FILE, "a", newline="", encoding="utf-8")
writer = csv.DictWriter(f, fieldnames=['location_id', 'temperature', 'humidity', 'wind_speed'])
if write_header:
    writer.writeheader()
    f.flush()

# 5) iterate missing ids and fetch one-by-one, appending on success
attempted = 0
succeeded = 0
skipped_no_coords = 0
failed_responses = []

for i, lid in enumerate(missing_ids, 1):
    # defensive: some ids in missing_ids may not actually exist in df_loc index if dirty
    try:
        row = df_loc.loc[df_loc['id'] == lid].iloc[0]
    except Exception:
        print(f"[{i}/{len(missing_ids)}] {lid}: not found in locations file, skipping")
        continue

    lat = row.get('latitude')
    lon = row.get('longitude')
    if pd.isna(lat) or pd.isna(lon):
        skipped_no_coords += 1
        print(f"[{i}/{len(missing_ids)}] {lid}: SKIP - missing coords")
        continue

    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    attempted += 1
    try:
        r = requests.get(url, timeout=12)
    except Exception as e:
        print(f"[{i}/{len(missing_ids)}] {lid}: EXC {e}")
        failed_responses.append((lid, f"EXC {e}"))
        time.sleep(1)
        continue

    if r.status_code == 200:
        j = r.json()
        out = {
            "location_id": int(lid),
            "temperature": j.get('main', {}).get('temp'),
            "humidity": j.get('main', {}).get('humidity'),
            "wind_speed": j.get('wind', {}).get('speed')
        }
        writer.writerow(out)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
        have_ids.add(int(lid))
        succeeded += 1
        print(f"[{i}/{len(missing_ids)}] {lid}: OK")
    else:
        print(f"[{i}/{len(missing_ids)}] {lid}: FAIL HTTP {r.status_code} - {r.text[:120]}")
        failed_responses.append((lid, f"HTTP {r.status_code}"))

    # small pause to be polite to API (adjust if you have higher quota)
    time.sleep(1)

# 6) close writer & final counts
f.close()
df_final = pd.read_csv(WEATHER_FILE)
print("Done.")
print(f"Requested missing: {len(missing_ids)}, attempted: {attempted}, succeeded: {succeeded}, skipped(no coords): {skipped_no_coords}")
print(f"Weather rows now: {len(df_final)}")

if failed_responses:
    print("\nSample failures (first 10):")
    for lid, reason in failed_responses[:10]:
        print(f"  {lid} -> {reason}")
    print("\nIf failures are HTTP 401/403 -> check API key; 429 -> rate limits; 404 -> bad coords.")
