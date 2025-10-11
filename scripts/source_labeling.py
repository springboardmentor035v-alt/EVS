import pandas as pd
import numpy as np

INPUT_FILE = "data/cleaned_features_new.csv"
OUTPUT_FILE = "data/labeled_data_new.csv"

# ----------------- LOAD DATA -----------------
df = pd.read_csv(INPUT_FILE)
print(f"🔹 Loaded cleaned dataset: {df.shape}")

# ----------------- SEASON SIMULATION -----------------
np.random.seed(42)
months = np.random.randint(1, 13, size=len(df))
df["month"] = months

def assign_season(lat, month):
    """Assign Dry/Wet season depending on latitude and month."""
    if abs(lat) < 15:  # Equatorial
        return "Dry" if month in [1,2,3,11,12] else "Wet"
    elif lat > 0:  # Northern Hemisphere
        return "Dry" if month in [3,4,5,6,7,8] else "Wet"
    else:  # Southern Hemisphere
        return "Dry" if month in [9,10,11,12,1,2] else "Wet"

df["season"] = df.apply(lambda r: assign_season(r["latitude"], r["month"]), axis=1)

# ----------------- LABELING HEURISTICS -----------------
no2_thresh = df["no2"].quantile(0.75)
so2_thresh = df["so2"].quantile(0.75)
pm10_thresh = df["pm10"].quantile(0.75)
pm25_thresh = df["pm2_5"].quantile(0.75)
co_thresh = df["co"].quantile(0.75)

def label_source(row):
    # Vehicular: high NO2 OR CO, with moderate/low wind
    if (row["no2"] > no2_thresh or row["co"] > co_thresh) and row["wind_speed"] < 6:
        return "Vehicular"
    # Industrial: high SO2 and PM10
    elif row["so2"] > so2_thresh and row["pm10"] > pm10_thresh * 0.8:
        return "Industrial"
    # Agricultural: high PM in dry season
    elif (row["pm10"] > pm10_thresh or row["pm2_5"] > pm25_thresh) and row["season"] == "Dry":
        return "Agricultural"
    # Burning: high CO or PM, with poor AQI
    elif (row["co"] > co_thresh or row["pm2_5"] > pm25_thresh) and row["aqi"] >= 3:
        return "Burning"
    # Default
    else:
        return "Natural"

df["pollution_source"] = df.apply(label_source, axis=1)

# ----------------- STATS & SAVE -----------------
print("🔹 Label distribution (counts):")
print(df["pollution_source"].value_counts())

print("\n🔹 Label distribution (percentages):")
print(df["pollution_source"].value_counts(normalize=True) * 100)

# Save labeled dataset (location_id and lat/lon preserved)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Labeled dataset saved to {OUTPUT_FILE}")
