🌍 EnviroScan: AI-Powered Pollution Source Identifier

EnviroScan is an AI-powered system for monitoring and identifying pollution sources across cities. It combines air quality data, weather information, and geospatial features with machine learning to classify pollution sources (Vehicular, Industrial, Agricultural, etc.) and provide real-time analytics, alerts, and interactive maps.

📖 Table of Contents

✨ Features

🛠️ Tech Stack

📂 Project Structure

⚙️ Installation

▶️ Running the Dashboard

📊 Dashboard Features

📌 Example Visuals

📜 Documentation

🙌 Collaborators

✨ Features

✅ Data Collection & Preprocessing

-- OpenAQ API for pollution data (PM2.5, PM10, NO₂, SO₂, O₃, CO).

-- OpenWeatherMap API for weather data.

-- OSMnx for extracting geospatial features.

-- Cleaned, normalized, and feature-engineered dataset.

✅ Machine Learning Model

-- Pollution source classification using Random Forest.

-- Confidence thresholding (predictions below 60% → “Uncertain”).

-- Saved model + scaler for reproducibility.

✅ Interactive Dashboard (Streamlit)

-- Multi-tab layout: Map | Analytics | Alerts | Predict.

-- Multi-select filters for cities and sources.

-- Heatmaps + marker clusters (Folium).

-- Pollutant trend charts (Plotly).

-- Alerts when AQI > 150.

-- Upload CSV → get predictions → download results.

✅ Deliverables

-- pollution_map.html – Interactive map.

-- EnviroScan_Dashboard – Streamlit app.

🛠️ Tech Stack

Python

Pandas, NumPy – Data processing

scikit-learn, joblib – Model training & inference

Streamlit, streamlit-folium, Plotly – Dashboard & visualization

Folium, OSMnx – Geospatial mapping

Requests – API integration

pdfkit – Report generation

📂 Project Structure EnviroScan/

│── data/

│ ├── pollution_data.csv

│ ├── specific_locations_cleaned.csv

│ ├── data_scaler.joblib

│── src/

│ ├── dashboard.py # Main Streamlit dashboard

│ ├── create_map.py # Folium map generation

│ ├── model_training.ipynb # Notebook for training & evaluation

│── pollution_source_model.joblib

│── requirements.txt

│── README.md

⚙️ Installation

Clone the repo and install dependencies:

git clone https://github.com/your-username/enviroScan.git

cd enviroScan

Create virtual environment (recommended)
python -m venv venv

source venv/bin/activate # Mac/Linux

venv\Scripts\activate # Windows

Install dependencies
pip install -r requirements.txt

Note: For PDF export, install wkhtmltopdf (system dependency):

sudo apt-get install wkhtmltopdf # Ubuntu/Debian

brew install wkhtmltopdf # Mac

▶️ Running the Dashboard

Run the Streamlit app:

streamlit run src/dashboard.py

Open your browser at http://localhost:8501 .

📊 Dashboard Features

🗺 Map Tab → Interactive Folium map with heatmaps and clustered pollution sources.

📊 Analytics Tab → Pie charts for source distribution + trend charts for pollutants.

⚠ Alerts Tab → Warnings when AQI > 150 with details of high-risk locations.

🤖 Predict Tab → Upload CSV → Get ML-based source predictions → Download results.

📌 Example Visuals

🌍 Pollution Sources Map

📊 Analytics Tab

⚠ Alerts

🙌 Collaborators

Gautham K

Kushal P Hiremath

Prasanna Venkatesh R
