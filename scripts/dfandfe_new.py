import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load Raw Dataset
# -------------------------------
df = pd.read_csv("data/merged_core_data.csv")
print("🔹 Raw shape:", df.shape)

# -------------------------------
# 2. Handle Missing Values
# -------------------------------
print("🔹 Missing values before:\n", df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Fill missing numerical values with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# -------------------------------
# 3. Feature Engineering
# -------------------------------
df["temp_humidity_index"] = df["temperature"] * df["humidity"] / 100
df["pollution_wind_ratio"] = (
    df["pm2_5"] + df["pm10"] + df["no2"] + df["so2"] + df["o3"] + df["co"]
) / (df["wind_speed"] + 0.1)

if "aqi" in df.columns:
    df["aqi_category"] = pd.cut(
        df["aqi"],
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=[
            "Good",
            "Moderate",
            "Unhealthy Sensitive",
            "Unhealthy",
            "Very Unhealthy",
            "Hazardous",
        ],
    )

# -------------------------------
# 4. Scale Only Numerical Features
# -------------------------------
scaler = StandardScaler()

# Define columns to scale (exclude location_id, lat/long, and keep locality/country untouched)
num_cols = [
    "pm2_5", "pm10", "no2", "so2", "o3", "co", "aqi",
    "temperature", "humidity", "wind_speed",
    "temp_humidity_index", "pollution_wind_ratio"
]

df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------------------
# 5. Keep Only Required Columns
# -------------------------------
final_cols = [
    "location_id", "pm2_5", "pm10", "no2", "so2", "o3", "co", "aqi",
    "temperature", "humidity", "wind_speed",
    "latitude", "longitude",
    "temp_humidity_index", "pollution_wind_ratio", "aqi_category",
    "locality", "country"
]

df = df[final_cols]

# -------------------------------
# 6. Save Final Clean Dataset
# -------------------------------
df.to_csv("data/cleaned_features_new.csv", index=False)
print("✅ Cleaned dataset saved as data/cleaned_features_new.csv")
