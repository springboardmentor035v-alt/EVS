# scripts/visualization_utils.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_agricultural_impact_chart(df):
    """
    Creates a visualization showing the relationship between 
    agricultural proximity and pollution levels.
    """
    if 'distance_to_nearest_agricultural_m' not in df.columns:
        return None
    
    # Create distance categories
    df_copy = df.copy()
    df_copy['agri_proximity'] = pd.cut(
        df_copy['distance_to_nearest_agricultural_m'],
        bins=[0, 1000, 2000, 5000, 10000, float('inf')],
        labels=['<1km', '1-2km', '2-5km', '5-10km', '>10km']
    )
    
    # Group by proximity and calculate average pollutants
    pollutants = ['pm25', 'pm10', 'no2', 'o3', 'co']
    available_pollutants = [p for p in pollutants if p in df_copy.columns]
    
    if not available_pollutants:
        return None
    
    grouped = df_copy.groupby('agri_proximity')[available_pollutants].mean().reset_index()
    
    # Create figure
    fig = go.Figure()
    
    for pollutant in available_pollutants:
        fig.add_trace(go.Bar(
            name=pollutant.upper(),
            x=grouped['agri_proximity'],
            y=grouped[pollutant],
            text=grouped[pollutant].round(2),
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Average Pollution Levels by Distance to Agricultural Areas",
        xaxis_title="Distance to Agriculture",
        yaxis_title="Concentration (µg/m³)",
        barmode='group',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_sub_area_comparison(df, pollutant='pm25'):
    """
    Creates a box plot comparing pollutant levels across sub-areas.
    """
    if 'sub_area' not in df.columns or pollutant not in df.columns:
        return None
    
    fig = px.box(
        df,
        x='sub_area',
        y=pollutant,
        color='sub_area',
        title=f"{pollutant.upper()} Distribution Across Sub-areas",
        labels={pollutant: f'{pollutant.upper()} (µg/m³)', 'sub_area': 'Sub-area'}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

def create_pollution_rose_chart(df, pollutant='pm25'):
    """
    Creates a wind rose style chart showing pollution levels by wind direction.
    """
    if 'wind_direction' not in df.columns or pollutant not in df.columns:
        return None
    
    # Create direction bins
    df_copy = df.copy()
    df_copy['direction_bin'] = pd.cut(
        df_copy['wind_direction'],
        bins=[-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 361],
        labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    )
    
    # Group by direction and calculate statistics
    grouped = df_copy.groupby('direction_bin')[pollutant].agg(['mean', 'count']).reset_index()
    
    fig = go.Figure(go.Barpolar(
        r=grouped['mean'],
        theta=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        width=[45]*8,
        marker_color=grouped['mean'],
        marker_colorscale='Reds',
        text=grouped['mean'].round(2),
        hovertemplate='<b>Direction: %{theta}</b><br>' +
                      f'{pollutant.upper()}: %{{r:.2f}} µg/m³<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, grouped['mean'].max() * 1.2]),
            angularaxis=dict(direction="clockwise")
        ),
        title=f"{pollutant.upper()} Pollution Rose (by Wind Direction)",
        showlegend=False
    )
    
    return fig

def create_source_timeline(df):
    """
    Creates a stacked area chart showing pollution source distribution over time.
    """
    if 'predicted_source' not in df.columns or 'timestamp' not in df.columns:
        return None
    
    # Group by date and source
    daily_sources = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'predicted_source']).size().reset_index(name='count')
    
    # Pivot for stacked area
    pivot_df = daily_sources.pivot(index='timestamp', columns='predicted_source', values='count').fillna(0)
    
    fig = go.Figure()
    
    colors = {
        'Vehicular': '#3498db',
        'Industrial': '#95a5a6',
        'Agricultural_Burning': '#e67e22',
        'Background_Mixed': '#27ae60'
    }
    
    for source in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[source],
            mode='lines',
            name=source,
            stackgroup='one',
            fillcolor=colors.get(source, '#000000'),
            line=dict(width=0.5, color=colors.get(source, '#000000'))
        ))
    
    fig.update_layout(
        title="Pollution Source Distribution Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Readings",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_multi_pollutant_radar(df, location_name=None):
    """
    Creates a radar chart comparing average levels of all pollutants.
    """
    pollutants = ['pm25', 'pm10', 'no2', 'o3', 'co', 'so2']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    if len(available_pollutants) < 3:
        return None
    
    # Calculate averages
    averages = df[available_pollutants].mean()
    
    # Normalize values for radar chart (0-100 scale based on thresholds)
    thresholds = {
        'pm25': 250, 'pm10': 300, 'no2': 400,
        'o3': 300, 'co': 50000, 'so2': 500
    }
    
    normalized = [(averages[p] / thresholds.get(p, averages[p])) * 100 
                  for p in available_pollutants]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=normalized,
        theta=[p.upper() for p in available_pollutants],
        fill='toself',
        name='Current Levels'
    ))
    
    # Add threshold line
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(available_pollutants),
        theta=[p.upper() for p in available_pollutants],
        fill='toself',
        name='Hazardous Level',
        line=dict(color='red', dash='dash')
    ))
    
    title_text = f"Pollutant Profile"
    if location_name:
        title_text += f" - {location_name}"
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 120]
            )
        ),
        showlegend=True,
        title=title_text
    )
    
    return fig

def create_correlation_scatter(df, x_pollutant='pm25', y_pollutant='no2'):
    """
    Creates a scatter plot showing correlation between two pollutants,
    colored by pollution source.
    """
    if x_pollutant not in df.columns or y_pollutant not in df.columns:
        return None
    
    color_col = 'predicted_source' if 'predicted_source' in df.columns else None
    
    fig = px.scatter(
        df,
        x=x_pollutant,
        y=y_pollutant,
        color=color_col,
        title=f"{x_pollutant.upper()} vs {y_pollutant.upper()} Correlation",
        labels={
            x_pollutant: f'{x_pollutant.upper()} (µg/m³)',
            y_pollutant: f'{y_pollutant.upper()} (µg/m³)'
        },
        trendline="ols",
        opacity=0.6
    )
    
    fig.update_layout(hovermode='closest')
    
    return fig

def create_hourly_heatmap(df, pollutant='pm25'):
    """
    Creates a heatmap showing pollution levels by hour and day of week.
    Requires hourly data.
    """
    if pollutant not in df.columns or 'timestamp' not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_of_week'] = df_copy['timestamp'].dt.day_name()
    
    # Pivot table
    heatmap_data = df_copy.pivot_table(
        values=pollutant,
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Reds',
        text=heatmap_data.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title=f"{pollutant.upper()}<br>(µg/m³)")
    ))
    
    fig.update_layout(
        title=f"{pollutant.upper()} Levels by Hour and Day of Week",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        xaxis=dict(dtick=1)
    )
    
    return fig

def calculate_air_quality_index(df):
    """
    Calculates a simple Air Quality Index based on multiple pollutants.
    Returns a dataframe with AQI scores.
    """
    aqi_df = df.copy()
    
    # Simple AQI calculation (normalized scale 0-500)
    pollutant_weights = {
        'pm25': 0.3,
        'pm10': 0.2,
        'no2': 0.2,
        'o3': 0.15,
        'co': 0.1,
        'so2': 0.05
    }
    
    thresholds = {
        'pm25': 250, 'pm10': 300, 'no2': 400,
        'o3': 300, 'co': 50000, 'so2': 500
    }
    
    aqi_df['AQI'] = 0
    
    for pollutant, weight in pollutant_weights.items():
        if pollutant in aqi_df.columns:
            normalized = (aqi_df[pollutant] / thresholds[pollutant]) * 500
            aqi_df['AQI'] += normalized * weight
    
    # Categorize AQI
    aqi_df['AQI_Category'] = pd.cut(
        aqi_df['AQI'],
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 
                'Unhealthy', 'Very Unhealthy', 'Hazardous']
    )
    
    return aqi_df