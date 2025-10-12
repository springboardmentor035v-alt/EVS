# EnviroScan: AI-Powered Pollution Source Identifier

 
*(Note: Replace the URL above with a screen recording/GIF of your final Streamlit app)*

## 📖 Overview

EnviroScan is an end-to-end data science project that identifies the likely sources of air pollution in major Indian cities using machine learning. The project culminates in an interactive Streamlit dashboard that allows users to explore historical pollution data and view AI-powered predictions on a geospatial map.

The model classifies pollution events into four primary categories:
-   🚗 **Vehicular:** Typically associated with high NO₂/CO levels near major roads.
-   🏭 **Industrial:** Characterized by high SO₂/PM10 concentrations near industrial zones.
-   🔥 **Agricultural Burning:** Linked to high PM2.5 levels near farmland, especially in dry conditions.
-   🌳 **Background/Mixed:** General atmospheric pollution without a single dominant source.

## ✨ Features

-   **Automated Data Pipeline:** Scripts to automatically collect historical air quality, weather, and geospatial data from multiple APIs.
-   **Simulated Ground Truth:** A robust data labeling process using heuristic rules and SMOTE to create a balanced training dataset.
-   **Machine Learning Model:** An XGBoost classifier trained and tuned to predict pollution sources with high accuracy.
-   **Interactive Dashboard:** A user-friendly web application built with Streamlit for data exploration and visualization.
-   **Geospatial Analysis:** An interactive Folium map displaying predicted pollution sources and pollutant concentration heatmaps.
-   **Data Visualization:** Dynamic charts from Plotly showing pollution trends and source distributions.

## 🛠️ Tech Stack

-   **Data Science & ML:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
-   **Data Collection:** OpenWeatherMap API, OSMnx (OpenStreetMap)
-   **Web App & Visualization:** Streamlit, Plotly, Folium
-   **Environment:** Python, Jupyter Notebooks

## 📂 Project Structure

```text
EnviroScan_Project/
|
|— app.py # Main Streamlit application file
|— data/
|   |— cities.csv # List of target cities for data collection
|   |— consolidated_enviro_data.csv # Raw, combined data from all sources
|   |— app_daily_data.csv # Pre-processed, aggregated data for the Streamlit app
|   |— train.csv, test.csv, validation.csv # Labeled & split datasets for ML
|   |— ...
|— models/
|   |— pollution_source_model.joblib # The final trained XGBoost model
|— notebooks/
|   |— 2_Data_Cleaning_and_Feature_Engineering.ipynb
|   |— 3_Source_Labeling_and_Simulation.ipynb
|   |— 4_Model_Training_and_Prediction.ipynb
|   |— 5_Geospatial_Visualization.ipynb
|— outputs/
|   |— *.html # Saved interactive maps
|— scripts/
|   |— config.py # Configuration for data collection
|   |— data_collector.py # Script to fetch all raw data
|   |— preprocess_for_app.py # Script to prepare data for the Streamlit app
|   |— ...
|— requirements.txt # Python dependencies
|— README.md # This file
```


## 🚀 Setup and Execution Guide

Follow these steps to set up the environment and run the entire project pipeline from data collection to deployment.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/EnviroScan_Project.git
cd EnviroScan_Project
```
2. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```bash
 # Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
3. Install Dependencies
Install all the required Python libraries.

```bash
pip install -r requirements.txt
```
4. Configure API Keys
-->1.The project requires an API key from OpenWeatherMap.
-->2.Get a free API key from OpenWeatherMap.
Open the scripts/config.py file and paste your key into the OPENWEATHER_API_KEY variable.
5. Full Pipeline Execution Workflow
Run the following steps in order. Note: Steps 5a and 5b can take a significant amount of time (potentially several hours) due to extensive data collection and processing.
a. Run the Data Collector
This script fetches several years of air quality and weather data for all cities listed in data/cities.csv.

```bash

cd scripts
python data_collector.py
cd ..
```
b. Run the Jupyter Notebooks
Execute the notebooks in numerical order to clean the data, create labels, and train the model.
Open and run all cells in notebooks/2_Data_Cleaning_and_Feature_Engineering.ipynb.
Open and run all cells in notebooks/3_Source_Labeling_and_Simulation.ipynb.
Open and run all cells in notebooks/4_Model_Training_and_Prediction.ipynb.
(Optional) Run notebooks/5_Geospatial_Visualization.ipynb to generate standalone map files.
After this step, your models/pollution_source_model.joblib file will be created.
c. Pre-process Data for the Web App
This script creates a smaller, aggregated file that allows the Streamlit app to load quickly.

```bash
cd scripts
python preprocess_for_app.py
cd ..
```

d. Launch the Streamlit Dashboard
You are now ready to run the final application!

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).
















