# scripts/03_label_and_split.py
import pandas as pd
from sklearn.model_selection import train_test_split
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def label_pollution_source(row):
    # Thresholds on NORMALIZED data (0-1 scale)
    NO2_HIGH, SO2_HIGH, PM25_HIGH = 0.6, 0.6, 0.5
    PM25_LOW, NO2_LOW, SO2_LOW = 0.2, 0.2, 0.2
    
    is_rush_hour = (7 <= row['hour'] <= 10) or (17 <= row['hour'] <= 20)
    is_dry_season = row['month'] in [3, 4, 5, 10, 11]

    # --- ENHANCED LOGIC ---
    # High-confidence rules first
    if row.get('so2', 0) > SO2_HIGH and row['industrial_count'] > 0.5:
        return 'Industrial'
    if row.get('no2', 0) > NO2_HIGH and is_rush_hour and row['roads_count'] > 0.5:
        return 'Vehicular'
    if row.get('pm2_5', 0) > PM25_HIGH and is_dry_season and row['agriculture_count'] > 0.5:
        return 'Agricultural Burning'
    
    # Rule for low pollution
    if row.get('pm2_5', 0) < PM25_LOW and row.get('no2', 0) < NO2_LOW and row.get('so2', 0) < SO2_LOW:
        return 'Natural/Low'
        
    return 'Mixed/Other'

def run_labeling_and_splitting():
    # ... (This script's main logic remains the same, only the labeling function is enhanced)
    logging.info("🚀 Starting Module 3: Source Labeling and Data Splitting")
    try:
        df = pd.read_csv(config.DATA_FOR_ML_FILE)
    except FileNotFoundError:
        logging.error(f"❌ Error: Input file not found. Run '02_data_processing.py' first.")
        return

    df[config.TARGET_COL] = df.apply(label_pollution_source, axis=1)
    df.to_csv(config.LABELED_FILE, index=False)
    logging.info(f"💾 Complete labeled dataset saved. Label distribution:\n{df[config.TARGET_COL].value_counts().to_string()}")

    try:
        train_df, test_df = train_test_split(
            df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df[config.TARGET_COL]
        )
    except ValueError as e:
        logging.error(f"\n❌ Error splitting data: {e}. Dataset might be too small or imbalanced.")
        return

    train_df.to_csv(config.TRAIN_FILE, index=False)
    test_df.to_csv(config.TEST_FILE, index=False)
    logging.info(f"💾 Train ({len(train_df)} rows) and test ({len(test_df)} rows) datasets saved.")
    logging.info("\n✅ Module 3 completed successfully!")

if __name__ == "__main__":
    run_labeling_and_splitting()