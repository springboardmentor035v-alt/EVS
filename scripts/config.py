# scripts/config.py (FINAL, ROBUST VERSION)
import os
from dotenv import load_dotenv
from pathlib import Path

# --- Build robust, absolute paths ---
# This ensures that no matter where you run the scripts from, the paths are always correct.
# PROJECT_ROOT will be the 'air_quality_project' directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the .env file from the project root
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')

# --- API Configuration ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- Directory and File Paths ---
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# Input
LOCATIONS_FILE = DATA_DIR / "locations.csv"

# Raw Data
AIR_QUALITY_FILE = OUTPUT_DIR / "raw_air_quality.csv"
WEATHER_FILE = OUTPUT_DIR / "raw_weather.csv"
OSM_FEATURES_FILE = OUTPUT_DIR / "raw_osm_features.csv"

# Processed Data
PROCESSED_EDA_FILE = OUTPUT_DIR / "processed_data_for_eda.csv"
DATA_FOR_ML_FILE = OUTPUT_DIR / "processed_data_for_ml.csv"
SCALER_FILE = OUTPUT_DIR / "scaler.joblib"

# Labeled Data
LABELED_FILE = OUTPUT_DIR / "labeled_data_for_dashboard.csv"
TRAIN_FILE = OUTPUT_DIR / "train_data.csv"
TEST_FILE = OUTPUT_DIR / "test_data.csv"

# Model Files
MODEL_FILE = OUTPUT_DIR / "pollution_source_model.joblib"
ENCODER_FILE = OUTPUT_DIR / "label_encoder.joblib"
EVALUATION_FILE = OUTPUT_DIR / "model_evaluation_report.txt"
CONFUSION_MATRIX_FILE = OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.png"

# --- Model Parameters ---
FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'temperature', 'humidity', 'wind_speed',
    'roads_count', 'industrial_count', 'agriculture_count', 'dumps_count',
    'co', 'nh3', 'no', 'no2', 'o3', 'pm10', 'pm2_5', 'so2'
]
TARGET_COL = 'pollution_source'
TEST_SIZE = 0.2
RANDOM_STATE = 42