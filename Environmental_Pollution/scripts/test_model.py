# scripts/test_model.py

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. DEFINE FILE PATHS (ROBUST METHOD) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, '..', 'models', 'pollution_source_model.joblib')
TEST_DATA_PATH = os.path.join(script_dir, '..', 'data', 'test.csv')
TRAIN_DATA_PATH = os.path.join(script_dir, '..', 'data', 'train.csv')


# --- 2. LOAD THE TRAINED MODEL AND SUPPORTING DATA ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {os.path.abspath(MODEL_PATH)}")

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop('pollution_source', axis=1)
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    y_train_labels = train_df['pollution_source']
    
    unique_labels = sorted(np.unique(y_train_labels))
    class_mapping = {label: i for i, label in enumerate(unique_labels)}
    inverse_class_mapping = {i: label for label, i in class_mapping.items()}
    print("\n✅ Class mapping created:")
    print(inverse_class_mapping)

except FileNotFoundError as e:
    print(f"❌ ERROR: Could not load a required file. Please check the path.")
    print(f"Attempted to load: {os.path.abspath(e.filename if hasattr(e, 'filename') else MODEL_PATH)}")
    exit()


# --- 3. METHOD 1: QUICK TEST WITH A SAMPLE FROM THE TEST SET ---
def test_with_sample_row():
    print("\n--- [ Method 1: Quick Test with a Sample Row ] ---")
    sample_row = X_test.iloc[[0]]
    prediction_encoded = model.predict(sample_row)
    prediction_label = inverse_class_mapping[prediction_encoded[0]]
    print(f"\nSample Data (first row of test set):")
    print(sample_row)
    print(f"\n🤖 Model Prediction: '{prediction_label}' (Encoded: {prediction_encoded[0]})")


# --- 4. METHOD 2: "REAL-WORLD" SIMULATION WITH CUSTOM DATA ---
def test_with_custom_data():
    print("\n--- [ Method 2: 'Real-World' Simulation ] ---")
    print("Simulating a high-traffic scenario: High NO2, near a major road.")

    custom_scenario = {
        'latitude': 28.7041, 'longitude': 77.1025,
        'co': 2500, 'no2': 95, 'o3': 20, 'pm10': 150, 'pm25': 80, 'so2': 15,
        'temperature': 35, 'humidity': 60, 'wind_speed': 2, 'wind_direction': 270,
        'distance_to_nearest_industrial_m': 4000,
        'distance_to_nearest_major_roads_m': 150,
        'distance_to_nearest_dump_site_m': 8000,
        'distance_to_nearest_agricultural_m': 10000,
        'is_weekend': 0,
        'hour_sin': np.sin(2 * np.pi * 9 / 24.0),
        'hour_cos': np.cos(2 * np.pi * 9 / 24.0),
        'month_sin': np.sin(2 * np.pi * 6 / 12.0),
        'month_cos': np.cos(2 * np.pi * 6 / 12.0),
        'location_Delhi_India': True,
        'location_Kolkata_India': False,
        'location_Mumbai_India': False
    }
    custom_df = pd.DataFrame([custom_scenario])

    # --- Preprocessing the custom data ---
    # 1. Define the numerical columns that need to be scaled.
    numerical_cols_for_scaling = [
        'co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'temperature', 'humidity',
        'wind_speed', 'wind_direction', 'distance_to_nearest_industrial_m',
        'distance_to_nearest_major_roads_m', 'distance_to_nearest_dump_site_m',
        'distance_to_nearest_agricultural_m'
    ]
    
    # 2. Load the TRAINING data to fit the scaler. This is crucial.
    #    We must scale our new data using the same scale as the data the model was trained on.
    X_train_for_scaler = train_df.drop('pollution_source', axis=1)
    scaler = StandardScaler()
    scaler.fit(X_train_for_scaler[numerical_cols_for_scaling])

    # 3. Scale the numerical features of our custom data point
    custom_df[numerical_cols_for_scaling] = scaler.transform(custom_df[numerical_cols_for_scaling])

    # 4. Ensure the column order is exactly the same as the training data
    custom_df = custom_df[X_test.columns]

    print("\nProcessed Custom Data (Scaled):")
    print(custom_df)

    # Make a prediction
    prediction_encoded = model.predict(custom_df)
    prediction_label = inverse_class_mapping[prediction_encoded[0]]

    print(f"\n🤖 Model Prediction for Custom Scenario: '{prediction_label}' (Encoded: {prediction_encoded[0]})")


# --- 5. RUN THE TESTS ---
if __name__ == "__main__":
    test_with_sample_row()
    print("\n" + "="*60 + "\n")
    test_with_custom_data()