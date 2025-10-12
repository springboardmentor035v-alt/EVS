#https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API key}
import requests
from pprint import pprint
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_current_weather(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data= response.json()
        return data["weather"][0].get("description")
    else:
        return {"error": "Unable to fetch weather data", "status_code": response.status_code}
    
# Example usage:
# api_key
if __name__ == "__main__":
    lat = 35.6895  # Example latitude for Tokyo
    lon = 139.6917 # Example longitude for Tokyo
    api_key = API_KEY
    weather_data = get_current_weather(lat, lon, api_key)
    pprint(weather_data)