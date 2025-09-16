import pandas as pd

data = [
    {"name": "Delhi", "lat": 28.6, "lon": 77.2, "temperature": 25, "humidity": 60, "wind_speed": 2.5, "timestamp": 1234567890, "source": "TestData"},
    {"name": "Mumbai", "lat": 19.0, "lon": 72.8, "temperature": 30, "humidity": 70, "wind_speed": 3.2, "timestamp": 1234567891, "source": "TestData"}
]

df = pd.DataFrame(data)
df.to_csv("outputs/test_weather.csv", index=False)
print("✅ Test weather data saved to outputs/test_weather.csv")
