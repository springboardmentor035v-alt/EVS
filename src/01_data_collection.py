import pandas as pd
import numpy as np

def simulate_data():
    np.random.seed(42)
    n = 200
    data = {
        "City": np.random.choice(["Hyderabad", "Mumbai", "Delhi", "Chennai"], n),
        "Latitude": np.random.uniform(12, 28, n),
        "Longitude": np.random.uniform(70, 85, n),
        "PM25": np.random.uniform(20, 200, n),
        "NO2": np.random.uniform(10, 80, n),
        "Wind_Speed": np.random.uniform(0.5, 5.0, n),
        "Temp": np.random.uniform(18, 40, n),
        "Pollution_Level": np.random.uniform(50, 300, n),
    }
    df = pd.DataFrame(data)
    df.to_csv("data/raw_data.csv", index=False)
    print("✅ Simulated raw data saved as data/raw_data.csv")

if __name__ == "__main__":
    simulate_data()
