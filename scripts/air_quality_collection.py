import requests
import pandas as pd
import os


locations_file = "data/locations.csv"
if not os.path.exists(locations_file):
    print(f"❌ File not found: {locations_file}")
    exit()

locations = pd.read_csv(locations_file)
if locations.empty:
    print("❌ locations.csv is empty!")
    exit()

API_KEY = "249cab13a0d583e15c55babcf0d8917d"
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"


data = []


for _, row in locations.iterrows():
    city = row["name"]
    lat = row["latitude"]
    lon = row["longitude"]
    
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            air = response.json()
            components = air["list"][0]["components"]
            timestamp = air["list"][0]["dt"]
            
            for param, value in components.items():
                data.append({
                    "name": city,
                    "lat": lat,
                    "lon": lon,
                    "parameter": param,
                    "value": value,
                    "unit": "µg/m³",
                    "timestamp": timestamp,
                    "source": "OpenWeatherMap AirPollution"
                })
            print(f"✅ Air quality fetched for {city}")
        else:
            print(f"⚠️ Failed for {city}, status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error for {city}: {e}")


output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "air_quality_data.csv")

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"✅ Air quality data saved to {output_file}")
