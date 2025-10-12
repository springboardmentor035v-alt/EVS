AI EnviroScan

AI EnviroScan is an AI-powered Air Quality Monitoring & Prediction Dashboard built with Streamlit.
It integrates historical data, machine learning models, and real-time APIs to provide insights on air quality in major Indian cities.

Project Features

1. Historical AQI Data

View historical AQI trends for selected cities.
Filter data by date range.
View latest AQI values with corresponding AQI Bucket.
Display pollution source distribution (e.g., Vehicles, Industry, Other Sources) based on pollutants.
Download historical data in CSV and PDF formats.

2. Future AQI Prediction

Predict AQI for selected cities using LSTM-based models.
Provides predicted AQI value, interval (confidence range), and AQI Bucket.
Simulates pollutant concentrations for the predicted date.
Input: future date. Output: AQI prediction and pollutant data.

3. Real-Time AQI

Fetches live AQI data using WAQI API.
Displays AQI value for selected city and station.
Shows pollutant concentrations in real-time.
Last updated timestamp provided.

4. AQI Bucket Classification

AQI is categorized into buckets:
Good (0–50)
Satisfactory (51–100)
Moderate (101–200)
Poor (201–300)
Very Poor (301–400)
Severe (401+)

5. Pollutants Handled

PM2.5, PM10, NO2, SO2, CO, O3, NO, NH3, Benzene, Toluene, Xylene
Pollutants are mapped to source categories like Vehicles, Industry, and Other Sources.

6. Visualization

Line charts for AQI trend over time.
Pie charts for pollutant source distribution.
Interactive maps using Folium for heatmap visualization.

7. Downloadable Reports

Historical AQI data can be downloaded in:
CSV format
PDF format with date, AQI, and bucket information.