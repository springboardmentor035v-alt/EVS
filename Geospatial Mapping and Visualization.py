import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd

def plot_enhanced_heatmap(df, date_range=None, location_filter=None, source_filter=None):
    # --- Apply Filters ---
    filtered_df = df.copy()
    if date_range:
        start, end = date_range
        filtered_df = filtered_df[(filtered_df['timestamp'] >= start) & (filtered_df['timestamp'] <= end)]
    if location_filter:
        lat_min, lat_max, lon_min, lon_max = location_filter
        filtered_df = filtered_df[
            (filtered_df['latitude'] >= lat_min) & (filtered_df['latitude'] <= lat_max) &
            (filtered_df['longitude'] >= lon_min) & (filtered_df['longitude'] <= lon_max)
        ]
    if source_filter:
        filtered_df = filtered_df[filtered_df['source'].isin(source_filter)]

    # --- Initialize Map ---
    m = folium.Map(location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom_start=8)

    # --- Heatmap Layer ---
    heat_data = [
        [row['latitude'], row['longitude'], row['PM2.5']]
        for _, row in filtered_df.iterrows() if not pd.isna(row['PM2.5'])
    ]
    HeatMap(heat_data, radius=18, blur=12, max_zoom=13, min_opacity=0.4).add_to(m)

    # --- Marker Cluster for Sources ---
    cluster = MarkerCluster().add_to(m)

    for _, row in filtered_df.iterrows():
        color = "orange"
        if row['source'] == "Industrial":
            color = "red"
        elif row['source'] == "Vehicular":
            color = "blue"
        elif row['source'] == "Agricultural":
            color = "green"

        popup_text = (
            f"<b>City:</b> {row['city']}<br>"
            f"<b>Source:</b> {row['source']}<br>"
            f"<b>PM2.5:</b> {row['PM2.5']}<br>"
            f"<b>NO2:</b> {row['NO2']}<br>"
            f"<b>SO2:</b> {row['SO2']}<br>"
            f"<b>Date:</b> {row['timestamp']}"
        )

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            fill_color=color,
            popup=popup_text
        ).add_to(cluster)

    # --- Add Legend ---
    legend_html = """
    <div style="position: fixed;
                bottom: 30px; left: 30px; width: 180px; height: 120px;
                border:2px solid grey; z-index:9999; font-size:14px;
                background:white; padding: 10px;">
        <b>Source Legend</b><br>
        <i style="background:red; width:10px; height:10px; float:left; margin-right:8px"></i> Industrial <br>
        <i style="background:blue; width:10px; height:10px; float:left; margin-right:8px"></i> Vehicular <br>
        <i style="background:green; width:10px; height:10px; float:left; margin-right:8px"></i> Agricultural <br>
        <i style="background:orange; width:10px; height:10px; float:left; margin-right:8px"></i> Other/Unknown <br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Save for export ---
    m.save("enhanced_pollution_map.html")
    return m
