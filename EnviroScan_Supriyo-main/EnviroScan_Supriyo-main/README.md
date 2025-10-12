🌱 Enviroscan7 – Environmental Data Analysis & Dashboard

Enviroscan7 is a Streamlit-based interactive dashboard designed to analyze, visualize, and monitor environmental air quality data.
It integrates OpenAQ, OpenWeather, OSM (OpenStreetMap) features, and Machine Learning models to detect pollution sources, generate insights, and issue real-time alerts.

Features

Data Upload & Processing – Upload environmental datasets (CSV) for analysis.
OSM Feature Extraction – Extracts road, industry, farm, and landfill information around a given location.
Weather Integration – Fetches temperature, humidity, pressure, and wind data via OpenWeather API.
Data Cleaning & Transformation – Fills missing values, standardizes features, and prepares ML-ready datasets.
Machine Learning Models
    Logistic Regression
    Random Forest
    Neural Network (MLPClassifier)

Visualizations

Pollutant trends over time
Pollution source distribution
Interactive Folium maps with heatmaps & markers


Real-Time Alerts – Detects when pollutants exceed thresholds and sends email notifications.

PDF Report Generation – Exports summarized AQI reports by station.


Tech Stack

Frontend: Streamlit, Folium, Seaborn, Matplotlib
Backend: Python, Pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE)
APIs: OpenAQ, OpenWeatherMap, OpenStreetMap (via OSMnx)
Alerts & Reports: ReportLab (PDF), SMTP (Email)


Project Structure

Enviroscan7/
app.py                  # Main Streamlit App
requirements.txt        # Dependencies
cleaned_environmental_data.csv  # Processed dataset (generated)
delhi_aq_data.csv/json  # Sample AQ dataset (generated)
delhi_meta_data.csv/json# Metadata (generated)
delhi_environmental_data.csv    # Consolidated dataset
README.md               # Project Documentation


Installation & Setup

Clone the Repository

git clone https://github.com/<your-username>/Enviroscan7.git
cd Enviroscan7

Install Dependencies

pip install -r requirements.txt

Run the Streamlit App

streamlit run app.py


Configuration

Update the following in app.py:

API Keys

OPENWEATHER_KEY = "your_openweather_api_key_here"


Email Alerts (SMTP Config)


EMAIL_CONFIG = {
    "sender": "your_email@gmail.com",
    "password": "your_app_password",
    "receiver": "receiver_email@gmail.com",
    "server": "smtp.gmail.com",
    "port": 587
}


Usage

1. Upload CSV: Upload an environmental dataset from OpenAQ or similar sources.
2. Select Location & Date Range: Enter city, latitude, longitude, and date filters.
3. Process Data: The app will clean, enrich, and prepare datasets.
4. Explore Dashboard:
       View pollutant trends
       Generate heatmaps
       Inspect pollution sources
       Download processed datasets
5. Alerts: If pollutants exceed thresholds → email notifications are sent automatically.
6. Reports: Export AQI summary reports in PDF format.


Example Thresholds

Pollutant	Threshold	Unit

PM2.5	50	µg/m³
PM10	100	µg/m³
NO₂	80	µg/m³
CO	10000	µg/m³
SO₂	75	µg/m³
O₃	70	µg/m³


Sample Dashboard

Trends & Distribution – Interactive pollutant plots
Map with Heatmap + Markers – Pollution source visualization
Alerts Panel – Real-time pollutant threshold checks



Notes

Ensure OpenWeather API key is valid.
For email alerts, you need to enable App Passwords (if using Gmail).
Large datasets may take longer to process.



License

This project is open-source under the MIT License.
