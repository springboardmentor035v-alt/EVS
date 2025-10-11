# scripts/merge_core_data.py
import pandas as pd
import ast

# Load CSVs
pollution = pd.read_csv("data/pollution_data.csv")
weather = pd.read_csv("data/weather_data.csv")
locations = pd.read_csv("data/global_locations_cleaned.csv")

# Merge pollution + weather
df = pollution.merge(weather, on="location_id", how="left")

# --- Helpers ---
def extract_lat_lon(coord_str):
    """Parse latitude/longitude from JSON-like string in coordinates column"""
    try:
        d = ast.literal_eval(coord_str)
        return pd.Series([d.get("latitude"), d.get("longitude")])
    except:
        return pd.Series([None, None])

# Extract lat/lon
locations[["latitude", "longitude"]] = locations["coordinates"].apply(extract_lat_lon)

# Get required columns from locations
loc = locations[["id", "name", "locality", "country", "latitude", "longitude"]].copy()

# Merge with df
df = df.merge(loc, left_on="location_id", right_on="id", how="left")

# Fill missing locality with name
df["locality"] = df["locality"].fillna(df["name"])

# --- Final selection ---
df_final = df[
    [
        "location_id",
        "pm2_5",
        "pm10",
        "no2",
        "so2",
        "o3",
        "co",
        "aqi",
        "temperature",
        "humidity",
        "wind_speed",
        "locality",
        "latitude",
        "longitude",
        "country",
    ]
]

# Save merged dataset
df_final.to_csv("data/merged_core_data.csv", index=False)
print("✅ Saved data/merged_core_data.csv with locality, lat/lon, and country.")
