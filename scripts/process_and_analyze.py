import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "outputs"
AIR_QUALITY_FILE = os.path.join(OUTPUT_DIR, "air_quality_data.csv")
WEATHER_FILE = os.path.join(OUTPUT_DIR, "weather_data.csv")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "physical_features.csv")
PROCESSED_FILE = os.path.join(OUTPUT_DIR, "processed_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)



def calculate_aqi_pm25(pm25):
    """
    Calculates the Air Quality Index (AQI) for PM2.5 based on Indian standards.
    This is a simplified version. A full implementation would check multiple pollutants.
    """
    if pd.isna(pm25):
        return np.nan
    if pm25 <= 30:
        return pm25 * 50 / 30
    elif pm25 <= 60:
        return 50 + (pm25 - 30) * 50 / 30
    elif pm25 <= 90:
        return 100 + (pm25 - 60) * 100 / 30
    elif pm25 <= 120:
        return 200 + (pm25 - 90) * 100 / 30
    elif pm25 <= 250:
        return 300 + (pm25 - 120) * 100 / 130
    else:
        return 400 + (pm25 - 250) * 100 / 130

def categorize_aqi(aqi):
    """Categorizes AQI value into human-readable labels."""
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def run_module_2():
    print("🚀 Starting Module 2: Data Cleaning, Feature Engineering & Analysis")

    try:
        air_df = pd.read_csv(AIR_QUALITY_FILE)
        weather_df = pd.read_csv(WEATHER_FILE)
        features_df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}. Make sure Module 1 ran successfully.")
        return

    merged_df = pd.merge(air_df, weather_df, on=['name', 'lat', 'lon'], how='outer', suffixes=('_air', '_weather'))
    final_df = pd.merge(merged_df, features_df, on=['name'], how='left', suffixes=('', '_features'))
    print("✅ Datasets merged successfully.")

    # === Step 2: Clean Datasets (CORRECTED ORDER) ===
    final_df.dropna(subset=['parameter', 'value'], inplace=True)
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp_air'], unit='s', errors='coerce')
    final_df['timestamp'].fillna(pd.to_datetime(final_df['timestamp_weather'], unit='s', errors='coerce'), inplace=True)
    final_df['latitude'] = final_df['lat'].fillna(final_df['latitude'])
    final_df['longitude'] = final_df['lon'].fillna(final_df['longitude'])
    cols_to_drop = [col for col in final_df.columns if '_air' in col or '_weather' in col or '_features' in col or col in ['lat', 'lon']]
    final_df.drop(columns=list(set(cols_to_drop)), inplace=True)
    numeric_cols = ['value', 'temperature', 'humidity', 'wind_speed', 'roads_count', 'industrial_count', 'agriculture_count', 'dumps_count']
    for col in numeric_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    feature_cols = ['roads_count', 'industrial_count', 'agriculture_count', 'dumps_count']
    final_df[feature_cols] = final_df[feature_cols].fillna(0)
    print("✅ Data cleaning complete.")

    df_wide = final_df.pivot_table(
        index=['name', 'latitude', 'longitude', 'timestamp', 'temperature', 'humidity', 'wind_speed', 'roads_count', 'industrial_count', 'agriculture_count', 'dumps_count'],
        columns='parameter',
        values='value'
    ).reset_index()

    print(f"\nℹ️  Available pollutant columns after pivot: {list(df_wide.columns)}")

    if 'pm2_5' in df_wide.columns:
        df_wide['aqi'] = df_wide['pm2_5'].apply(calculate_aqi_pm25)
        df_wide['aqi_category'] = df_wide['aqi'].apply(categorize_aqi)
        print("✅ AQI calculated and categorized from 'pm2_5'.")
    else:
        print("⚠️  Warning: 'pm2_5' column not found. Skipping AQI calculation.")
        df_wide['aqi'] = np.nan
        df_wide['aqi_category'] = 'Unknown'

    search_area_km2 = np.pi * (2**2)
    df_wide['road_density'] = df_wide['roads_count'] / search_area_km2
    print("✅ Derived features (road_density) created.")


    plt.figure(figsize=(12, 8))
    corr_cols = ['aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'temperature', 'humidity', 'wind_speed', 'road_density', 'industrial_count']
    corr_cols_exist = [col for col in corr_cols if col in df_wide.columns and df_wide[col].notna().any()]
    
    if len(corr_cols_exist) > 1:
        correlation_matrix = df_wide[corr_cols_exist].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Between Pollutants, Weather, and Physical Features')
        correlation_plot_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
        plt.savefig(correlation_plot_path)
        plt.close()
        print(f"📊 Correlation heatmap saved to {correlation_plot_path}")
    else:
        print("⚠️  Warning: Not enough data to create a correlation heatmap.")

    if df_wide['aqi'].notna().any():
        avg_aqi_by_city = df_wide.groupby('name')['aqi'].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        avg_aqi_by_city.plot(kind='bar', color=sns.color_palette("viridis", len(avg_aqi_by_city)))
        plt.title('Average AQI Comparison by City')
        plt.ylabel('Average AQI')
        plt.xlabel('City')
        plt.xticks(rotation=45)
        plt.tight_layout()
        city_comparison_path = os.path.join(OUTPUT_DIR, 'city_aqi_comparison.png')
        plt.savefig(city_comparison_path)
        plt.close()
        print(f"📊 City AQI comparison chart saved to {city_comparison_path}")
    else:
        print("⚠️  Warning: No AQI data available. Skipping city comparison chart.")


    df_wide.to_csv(PROCESSED_FILE, index=False)
    print(f"💾 Fully processed data saved to {PROCESSED_FILE}")
    print("\nFinal Processed Data Sample:")
    print(df_wide.head())



if __name__ == "__main__":
    run_module_2()