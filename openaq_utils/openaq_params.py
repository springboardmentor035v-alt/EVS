import os
import requests
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}




def get_params(required_params=None):
    """
    Fetch OpenAQ parameters and return id & units for selected parameters.


    Args:
    required_params (list): list of parameter names (e.g. ['pm25','pm10',...])


    Returns:
    dict: {param_name: {'id': id, 'units': units} } (if not found -> None)
    """
    if required_params is None:
     required_params = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']


    url = "https://api.openaq.org/v3/parameters"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()


    results = resp.json().get('results', [])
    param_info = {}


    for item in results:
        name = item.get('name')
        if name in required_params:
            units = item.get('units')
            # keep first, but prefer µg/m³ if multiple unit variants appear
            if name not in param_info:
                param_info[name] = {'id': item.get('id'), 'units': units}
            else:
            # replace if current stored unit is not µg/m³ and this one is µg/m³
                if param_info[name].get('units') != 'µg/m³' and units == 'µg/m³':
                    param_info[name] = {'id': item.get('id'), 'units': units}


    # Ensure keys exist for all requested params (value None if absent)
    for p in required_params:
         param_info.setdefault(p, None)


    return param_info


# Example usage:            
if __name__ == "__main__":
    params = get_params(['pm25', 'pm10', 'no2', 'co', 'so2', 'o3', 'bc'])
    for p, info in params.items():
        if info:
            print(f"{p}: id={info['id']}, units={info['units']}")
        else:
            print(f"{p}: Not found")