# scripts/config.py

# --- API Keys and Configuration ---
# NOTE: We are now using the OpenAQ API for air quality. 
# The free OpenWeatherMap key is still used for weather data.
OPENWEATHER_API_KEY = "f893c37eda21476de86bad7c65e10e10"
# OpenAQ does not require an API key for public data access.

# --- Directory and File Configuration ---
DATA_DIR = "../data/"
CITIES_FILE = "../data/cities.csv" # New file to list target cities

# --- Data Collection Parameters ---
# Pollutants to be collected from OpenAQ.
POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]

# --- Geospatial Analysis Parameters ---
# Radius in meters around the target coordinates to search for features.
GEOSPATIAL_RADIUS_METERS = 5000  # 5 kilometers

# OpenStreetMap feature tags to extract.
OSM_TAGS = {
    "industrial": {"landuse": "industrial"},
    "major_roads": {"highway": ["primary", "secondary", "motorway"]},
    "dump_site": {"landuse": "landfill"},
    "agricultural": {"landuse": ["farmland", "farmyard"]}
}