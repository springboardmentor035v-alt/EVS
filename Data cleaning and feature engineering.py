import pandas as pd
import numpy as np

# --- Enhanced Data Cleaning ---
def clean_collected_data(df):
    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Ensure numeric fields
    numeric_cols = ['latitude', 'longitude', 'temperature', 'humidity', 'roads_count', 'factories_count']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Handle missing values
    # - Critical columns: drop rows with missing city or coords
    df = df.dropna(subset=['city', 'latitude', 'longitude'])

    # - For other numeric columns: fill missing with median (more robust than mean)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # - For categorical/text columns: fill missing with "Unknown"
    cat_cols = ['weather_condition'] # Changed from 'weather_desc' to 'weather_condition'
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    return df


# --- Enhanced Feature Engineering ---
def feature_engineering(df):
    # 1. Standardize numeric features (z-score normalization)
    for col in ['temperature', 'humidity', 'roads_count', 'factories_count']:
        df[col + "_zscore"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)  # avoid divide by zero

    # 2. Pollution complexity feature: count how many pollutants were measured
    # Assuming pollution data is stored in columns like 'pm25', 'pm10', etc.
    # We can count non-null values in these columns.
    pollution_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    df['pollution_count'] = df[pollution_cols].notna().sum(axis=1)


    # 3. Weather category: hot / moderate / cold
    df['weather_category'] = pd.cut(df['temperature'],
                                    bins=[-50, 15, 30, 60],
                                    labels=['Cold', 'Moderate', 'Hot'])

    # 4. Infrastructure density score (roads + factories)
    df['infra_score'] = np.log1p(df['roads_count']) + np.log1p(df['factories_count'])

    return df


# --- Run on collected_data.csv ---
df = pd.read_csv("collected_data_clean.csv")

# Clean
df_clean = clean_collected_data(df)

# Feature Engineering
df_final = feature_engineering(df_clean)

# Save cleaned dataset
df_final.to_csv("collected_data_cleaned.csv", index=False)

print("✅ Cleaned & enhanced data saved to collected_data_cleaned.csv")
