import os
import requests
from dotenv import load_dotenv
from math import radians, sin, cos, sqrt, atan2


load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def get_sensors(lat, lon, radius=1000, limit=10):
            """
            Query /v3/locations for stations near the given lat/lon.


            Returns a dict keyed by station id with values:
            { 'station_name', 'coordinates', 'distance_m', 'sensors': [ {sensor_id, parameter, units}, ... ] }
            """
            url = "https://api.openaq.org/v3/locations"
            params = {"coordinates": f"{lat},{lon}", "radius": radius, "limit": limit}
            resp = requests.get(url, headers=HEADERS, params=params)
            resp.raise_for_status()


            results = resp.json().get('results', [])
            stations = {}


            for st in results:
                    coords = st.get('coordinates') or {}
                    st_lat = coords.get('latitude')
                    st_lon = coords.get('longitude')
                    dist = None
                    if st_lat is not None and st_lon is not None:
                        dist = _haversine_m(lat, lon, st_lat, st_lon)


                    sensors = []
                    for s in st.get('sensors', []):
                        param = s.get('parameter', {})
                        sensors.append({
                        'sensor_id': s.get('id'),
                        'parameter': param.get('name'),
                        'units': param.get('units')
                        })


                    stations[st.get('id')] = {
                    'station_name': st.get('name'),
                    'coordinates': coords,
                    'distance_m': dist,
                    'sensors': sensors
                    }


            return stations