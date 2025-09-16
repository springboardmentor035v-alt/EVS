import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress the FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Configuration ---
OUTPUT_DIR = "outputs"
AIR_QUALITY_FILE = os.path.join(OUTPUT_DIR, "air_quality_data.csv")
WEATHER_FILE = os.path.join(OUTPUT_DIR, "weather_data.csv")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "physical_features.csv")
PROCESSED_FILE = os.path.join(OUTPUT_DIR, "processed_data_for_eda.csv")
ML_READY_FILE = os.path.join(OUTPUT_DIR, "normalized_data_for_ml.csv")


def get_season(month):
    """Assigns a season based on the month for the Indian context."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else: # 10, 11
        return 'Post-Monsoon'

def run_advanced_module_2():
    print("🚀 Starting Advanced Module 2: ML Data Preparation")

    # --- Step 1: Combine Datasets ---
    try:
        air_df = pd.read_csv(AIR_QUALITY_FILE)
        weather_df = pd.read_csv(WEATHER_FILE)
        features_df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Ensure Module 1 scripts have run successfully.")
        return

    # Merge into a unified DataFrame
    df = pd.merge(air_df, weather_df, on=['name', 'lat', 'lon'], how='outer', suffixes=('_air', '_weather'))
    df = pd.merge(df, features_df, on=['name'], how='left', suffixes=('', '_features'))
    print("✅ Datasets combined successfully.")

    # --- Step 2: Clean, Handle Missing Values, and Standardize ---
    # Remove duplicate entries
    df.drop_duplicates(inplace=True)
    # Remove records where essential data (pollutant value) is invalid/missing
    df.dropna(subset=['parameter', 'value'], inplace=True)

    # Standardize timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp_air'], unit='s', errors='coerce')
    df['timestamp'].fillna(pd.to_datetime(df['timestamp_weather'], unit='s', errors='coerce'), inplace=True)
    df.dropna(subset=['timestamp'], inplace=True) # Drop if timestamp is still null

    # Standardize GPS coordinates
    df['latitude'] = df['lat'].fillna(df['latitude'])
    df['longitude'] = df['lon'].fillna(df['longitude'])

    # Clean up old/redundant columns
    cols_to_drop = [col for col in df.columns if '_air' in col or '_weather' in col or '_features' in col or col in ['lat', 'lon']]
    df.drop(columns=list(set(cols_to_drop)), inplace=True)

    # Handle missing values for features (mean/median imputation or fill)
    feature_cols = ['roads_count', 'industrial_count', 'agriculture_count', 'dumps_count']
    df[feature_cols] = df[feature_cols].fillna(0) # Filling with 0 is reasonable if no features were found

    # Impute missing weather data with the median for robustness
    for col in ['temperature', 'humidity', 'wind_speed']:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    print("✅ Duplicates removed, missing values handled, and data standardized.")


    # --- Step 3: Feature Engineering (Temporal & Spatial) ---
    # Derive temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].apply(get_season)
    print("✅ Temporal features (hour, day, season) derived.")

    # Pivot data to wide format to work with pollutants as columns
    df_wide = df.pivot_table(
        index=['name', 'latitude', 'longitude', 'timestamp', 'hour', 'day_of_week', 'month', 'season',
               'temperature', 'humidity', 'wind_speed', 'roads_count', 'industrial_count',
               'agriculture_count', 'dumps_count'],
        columns='parameter',
        values='value'
    ).reset_index()

    # Calculate spatial proximity features (as a proxy)
    # Note: A true distance calculation requires coordinates of each feature.
    # We use an inverse relationship as a smart proxy: more features nearby = higher proximity value.
    for feature in ['roads', 'industrial', 'agriculture', 'dumps']:
        count_col = f'{feature}_count'
        if count_col in df_wide.columns:
            # Add 1 to avoid division by zero
            df_wide[f'{feature}_proximity'] = 1 / (1 + df_wide[count_col])
    print("✅ Spatial proximity features calculated (as a proxy).")

    # Calculate AQI
    if 'pm2_5' in df_wide.columns:
        # Impute missing pollutant values before calculating AQI
        df_wide['pm2_5'].fillna(df_wide['pm2_5'].median(), inplace=True)
        df_wide['aqi'] = df_wide['pm2_5'].apply(calculate_aqi_pm25) # Using a simplified AQI function
    else:
        df_wide['aqi'] = np.nan
    print("✅ AQI calculated.")

    # --- Step 4: Normalize Data for Model Input ---
    # Select only the numeric columns for scaling
    numeric_cols = df_wide.select_dtypes(include=np.number).columns.tolist()
    # Exclude identifiers from scaling
    cols_to_exclude = ['latitude', 'longitude']
    cols_to_scale = [col for col in numeric_cols if col not in cols_to_exclude]

    # Create a copy for the normalized data
    df_ml = df_wide.copy()

    # Initialize and apply the scaler
    scaler = MinMaxScaler()
    df_ml[cols_to_scale] = scaler.fit_transform(df_ml[cols_to_scale])
    print("✅ Pollutant and weather values normalized for ML.")

    # --- Step 5: Save the Unified, Feature-Rich DataFrames ---
    # Save the human-readable, un-normalized data for analysis and visualization
    df_wide.to_csv(PROCESSED_FILE, index=False)
    print(f"💾 Human-readable data for EDA saved to '{PROCESSED_FILE}'")

    # Save the normalized, ML-ready data for the next module
    df_ml.to_csv(ML_READY_FILE, index=False)
    print(f"💾 Normalized, ML-ready data saved to '{ML_READY_FILE}'")
    print("\n✅ Advanced Module 2 finished successfully!")


# This block makes the script runnable
if __name__ == "__main__":
    # Import the simplified AQI functions from the other script, or define them here
    from process_and_analyze import calculate_aqi_pm25
    run_advanced_module_2()