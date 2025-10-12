import folium
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
from typing import List, Dict, Any, Optional
from io import BytesIO
import base64

try:
    from config.settings import SOURCE_COLORS, MAP_CONFIG
except ImportError:
    # Fallback settings if config module not available
    SOURCE_COLORS = {
        "industrial": "#FF6B6B", "vehicular": "#4ECDC4", "agricultural": "#45B7D1",
        "urban_high": "#96CEB4", "urban_moderate": "#FECA57", 
        "background": "#FF9FF3", "clean": "#54A0FF"
    }
    MAP_CONFIG = {
        "default_location": [20.0, 78.0], "default_zoom": 5,
        "heatmap_radius": 18, "heatmap_blur": 15, "heatmap_min_opacity": 0.3
    }

class VisualizationEngine:
    """Advanced visualization engine for pollution data"""
    
    @staticmethod
    def create_pollution_map(df: pd.DataFrame, 
                           center: Optional[List[float]] = None,
                           zoom: Optional[int] = None) -> folium.Map:
        """Create an interactive pollution map"""
        if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
            return folium.Map(
                location=MAP_CONFIG["default_location"],
                zoom_start=MAP_CONFIG["default_zoom"]
            )
        
        # Calculate center if not provided
        if center is None:
            center = [df['latitude'].mean(), df['longitude'].mean()]
        if zoom is None:
            zoom = 10 if len(df) > 1 else MAP_CONFIG["default_zoom"]
        
        m = folium.Map(location=center, zoom_start=zoom)
        
        # Add heatmap if we have PM2.5 data
        if 'pm25' in df.columns:
            heat_data = df[['latitude', 'longitude', 'pm25']].dropna().values.tolist()
            if heat_data:
                HeatMap(
                    heat_data,
                    radius=MAP_CONFIG["heatmap_radius"],
                    blur=MAP_CONFIG["heatmap_blur"],
                    min_opacity=MAP_CONFIG["heatmap_min_opacity"]
                ).add_to(m)
        
        # Add markers by source type
        if 'predicted_source' in df.columns:
            for source in df['predicted_source'].unique():
                source_group = folium.FeatureGroup(name=f"Source: {source}")
                source_data = df[df['predicted_source'] == source]
                
                for _, row in source_data.iterrows():
                    # Create popup text
                    popup_text = f"""
                    <div style='font-family: Arial; font-size: 12px;'>
                    <b>Source:</b> {row.get('predicted_source', 'N/A')}<br>
                    <b>PM2.5:</b> {row.get('pm25', 'N/A')}<br>
                    <b>Confidence:</b> {row.get('confidence', 0):.2f if pd.notna(row.get('confidence')) else 'N/A'}<br>
                    <b>Lat:</b> {row['latitude']:.4f}, <b>Lon:</b> {row['longitude']:.4f}
                    </div>
                    """
                    
                    color = SOURCE_COLORS.get(str(row.get('predicted_source')), '#000000')
                    pm25_value = row.get('pm25', 0)
                    
                    # Calculate marker size based on PM2.5
                    base_radius = 6
                    if 'pm25' in df.columns and df['pm25'].max() > 0:
                        radius = base_radius + (pm25_value / df['pm25'].max()) * 10
                    else:
                        radius = base_radius
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_text, max_width=300)
                    ).add_to(source_group)
                
                source_group.add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    
    @staticmethod
    def create_pollution_dashboard_charts(df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """Create multiple pollution analysis charts"""
        charts = {}
        
        try:
            # Source distribution pie chart
            if 'predicted_source' in df.columns:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                source_counts = df['predicted_source'].value_counts()
                if not source_counts.empty:
                    colors = [SOURCE_COLORS.get(str(src), '#CCCCCC') for src in source_counts.index]
                    ax1.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', colors=colors)
                    ax1.set_title('Pollution Source Distribution')
                    charts['source_distribution'] = fig1
            
            # Pollutant levels bar chart
            pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
            available_pollutants = [p for p in pollutants if p in df.columns]
            
            if available_pollutants:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                pollutant_data = {p: df[p].mean() for p in available_pollutants}
                ax2.bar(pollutant_data.keys(), pollutant_data.values(), color='skyblue')
                ax2.set_title('Average Pollutant Levels')
                ax2.set_ylabel('Concentration (μg/m³)')
                plt.xticks(rotation=45)
                charts['pollutant_levels'] = fig2
            
            # Time series trend if date column exists
            if 'date' in df.columns:
                try:
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    df_temp = df.copy()
                    df_temp['date'] = pd.to_datetime(df_temp['date'])
                    daily_avg = df_temp.groupby(df_temp['date'].dt.date)['pm25'].mean()
                    ax3.plot(daily_avg.index, daily_avg.values, marker='o', linewidth=2, color='red')
                    ax3.set_title('Daily Average PM2.5 Trend')
                    ax3.set_xlabel('Date')
                    ax3.set_ylabel('PM2.5 (μg/m³)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    charts['time_series'] = fig3
                except Exception as e:
                    print(f"Time series chart error: {e}")
        
        except Exception as e:
            print(f"Chart creation error: {e}")
        
        return charts
    
    @staticmethod
    def create_single_prediction_chart(prediction_data: Dict[str, Any]) -> plt.Figure:
        """Create visualization for single prediction"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence gauge
        confidence = prediction_data.get('confidence', 0)
        ax1.barh(['Confidence'], [confidence], color='green' if confidence > 0.7 else 'orange')
        ax1.set_xlim(0, 1)
        ax1.set_title('Prediction Confidence')
        ax1.set_xlabel('Confidence Score')
        
        # Pollutant levels vs thresholds
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        levels = [prediction_data.get(p, 0) for p in pollutants]
        thresholds = [100, 200, 200, 100, 9, 180]  # Default thresholds
        
        x = range(len(pollutants))
        ax2.bar(x, levels, color=['red' if l > t else 'blue' for l, t in zip(levels, thresholds)])
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.upper() for p in pollutants], rotation=45)
        ax2.set_title('Pollutant Levels vs Thresholds')
        ax2.set_ylabel('Concentration')
        
        plt.tight_layout()
        return fig