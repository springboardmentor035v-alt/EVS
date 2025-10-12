import os
import requests
from dotenv import load_dotenv
from openaq_utils.openaq_sensors import get_sensors
from weather_utils.current_weather import get_current_weather   # import your weather function

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")  # store weather key in .env too
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

def get_values(lat, lon, required_params=None, radius=5000):
    """
    For given lat/lon return a single dict suitable for creating one DataFrame row.

    Keys: required parameter names (values or None), plus 
    'latitude','longitude','station_id','station_name','weather'.
    """
    if required_params is None:
        required_params = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']

    stations = get_sensors(lat, lon, radius=radius, limit=10)
    if not stations:
        row = {p: None for p in required_params}
        row.update({
            'latitude': lat,
            'longitude': lon,
            'station_id': None,
            'station_name': None,
            'weather': get_current_weather(lat, lon, WEATHER_KEY)  # ✅ add weather here
        })
        return row

    # pick nearest station
    nearest_id, nearest = min(
        stations.items(),
        key=lambda kv: (kv[1].get('distance_m') if kv[1].get('distance_m') is not None else float('inf'))
    )

    row = {p: None for p in required_params}
    row.update({
        'latitude': lat,
        'longitude': lon,
        'station_id': nearest_id,
        'station_name': nearest.get('station_name'),
        'weather': get_current_weather(lat, lon, WEATHER_KEY)  # ✅ add weather here
    })

    for s in nearest.get('sensors', []):
        param = s.get('parameter')
        if not param or param not in required_params:
            continue
        sid = s.get('sensor_id')
        meas_url = f"https://api.openaq.org/v3/sensors/{sid}/measurements"
        r = requests.get(meas_url, headers=HEADERS, params={"limit": 1, "sort": "desc"})
        if r.status_code != 200:
            continue
        mvals = r.json().get('results', [])
        if mvals:
            row[param] = mvals[0].get('value')



    return row
