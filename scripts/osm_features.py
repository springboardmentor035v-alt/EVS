import os
import pandas as pd
import osmnx as ox


os.makedirs("outputs", exist_ok=True)

# Load locations
locations = pd.read_csv("data/locations.csv")

all_features = []

for _, row in locations.iterrows():
    point = (row["latitude"], row["longitude"])
    
    G = ox.graph_from_point(point, dist=2000, network_type='all')
    roads_count = len(G.edges)
    

    try:
        industrial = ox.features_from_point(point, tags={'landuse':'industrial'}, dist=2000)
        industrial_count = len(industrial)
    except ox._errors.InsufficientResponseError:
        industrial_count = 0
    
    # Agricultural fields
    try:
        agriculture = ox.features_from_point(point, tags={'landuse':'farmland'}, dist=2000)
        agriculture_count = len(agriculture)
    except ox._errors.InsufficientResponseError:
        agriculture_count = 0
    
    # Dump sites / landfill
    try:
        dumps = ox.features_from_point(point, tags={'landuse':'landfill'}, dist=2000)
        dumps_count = len(dumps)
    except ox._errors.InsufficientResponseError:
        dumps_count = 0
    
    all_features.append({
        "name": row["name"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "roads_count": roads_count,
        "industrial_count": industrial_count,
        "agriculture_count": agriculture_count,
        "dumps_count": dumps_count
    })


df = pd.DataFrame(all_features)
df.to_csv("outputs/physical_features.csv", index=False)
print("✅ Physical features saved to outputs/physical_features.csv")
