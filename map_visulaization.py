import pandas as pd
import folium
from folium.plugins import HeatMap

# Load your preprocessed dataset
df = pd.read_csv("./data/pollution_source_classification_dataset.csv")

# Create base map (centered on mean coordinates)
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)

# 1. Heatmap layer (use PM2.5 as severity)
heat_data = df[['latitude', 'longitude', 'pm25']].values.tolist()
HeatMap(heat_data, radius=20).add_to(m)

# 2. Markers for predicted sources
for _, row in df.iterrows():
    popup_text = (
        f"Source: {row['pollution_source']}<br>"
        f"PM2.5: {row['pm25']}<br>"
        f"PM10: {row['pm10']}<br>"
        f"NO2: {row['no2']}"
    )
    
    # Simple color rule
    if row['pollution_source'] == "industrial":
        color = "red"
    elif row['pollution_source'] == "vehicular":
        color = "blue"
    else:
        color = "green"
    
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup_text,
        icon=folium.Icon(color=color)
    ).add_to(m)

# 3. Add Legend for pollution sources (bottom-left)
legend_html = """
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; width: 150px; height: 120px; 
    background-color: white; 
    border:2px solid grey; z-index:9999; font-size:14px;
    padding: 10px;
">
<b>Pollution Sources</b><br>
<i style="color:red" class="fa fa-map-marker"></i> Industrial<br>
<i style="color:blue" class="fa fa-map-marker"></i> Vehicular<br>
<i style="color:green" class="fa fa-map-marker"></i> Other Sources<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# 4. Add Heatmap color scale (bottom-right)
color_scale_html = """
<div style="
    position: fixed; 
    bottom: 50px; right: 50px; width: 160px; height: 80px; 
    background-color: white; 
    border:2px solid grey; z-index:9999; font-size:14px;
    padding: 10px;
">
<b>PM2.5 Scale</b><br>
<span style="background:blue; width:20px; display:inline-block;">&nbsp;</span> Low<br>
<span style="background:red; width:20px; display:inline-block;">&nbsp;</span> High
</div>
"""
m.get_root().html.add_child(folium.Element(color_scale_html))

# Save map to HTML
m.save("pollution_map.html")
print(" Map with legend + scale bar saved as pollution_map.html")
