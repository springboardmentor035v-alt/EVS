import pandas as pd
import os


outputs_dir = "outputs"
processed_file = os.path.join(outputs_dir, "processed_data.csv")

air_file = os.path.join(outputs_dir, "air_quality_data.csv")
weather_file = os.path.join(outputs_dir, "weather_data.csv")
features_file = os.path.join(outputs_dir, "physical_features.csv")

air_df = pd.read_csv(air_file)
weather_df = pd.read_csv(weather_file)
features_df = pd.read_csv(features_file)

df = pd.merge(
    air_df, 
    weather_df, 
    on='name', 
    how='outer', 
    suffixes=('_air', '_weather')
)

df = pd.merge(
    df, 
    features_df, 
    on='name', 
    how='left'
)

print(f"✅ Merged data loaded, shape: {df.shape}")


numeric_cols = ["value", "temperature", "humidity", "wind_speed", 
                "roads_count", "industrial_count", "agriculture_count", "dumps_count"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
     
        if col in ["roads_count", "industrial_count", "agriculture_count", "dumps_count"]:
            df[col] = df[col].fillna(0)

if "value" in df.columns:
    df = df.dropna(subset=["value"])


for col in ["timestamp_air", "timestamp_weather"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')


df = df.drop_duplicates().reset_index(drop=True)

df.to_csv(processed_file, index=False)
print(f"✅ Cleaned & processed data saved to {processed_file}")
print(df.head())
