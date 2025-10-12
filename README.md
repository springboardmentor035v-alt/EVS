
---

````markdown
# 🌍 EnviroScan: Global Air Pollution and Weather Analysis Dashboard

**Live Dashboard:** 👉 [https://enviroscan-keerthi.streamlit.app/](https://enviroscan-keerthi.streamlit.app/)

EnviroScan is a global air quality monitoring and analysis platform that integrates **real-time air pollution** and **weather data** from multiple APIs.  
It performs **data cleaning, feature engineering, machine learning–based pollution source prediction**, and provides **interactive global visualizations** through a Streamlit dashboard.

---

## 📘 Table of Contents
1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Project Workflow](#-project-workflow)
4. [Dataset Description](#-dataset-description)
5. [Setup Instructions](#-setup-instructions)
6. [Script Descriptions](#-script-descriptions)
7. [Running the Pipeline](#-running-the-pipeline)
8. [Model Training and Outputs](#-model-training-and-outputs)
9. [Streamlit Dashboard](#-streamlit-dashboard)
10. [Directory Structure](#-directory-structure)
11. [Future Enhancements](#-future-enhancements)
12. [Contributors](#-contributors)
13. [License](#-license)

---

## 🌫 Project Overview

Air pollution affects nearly every region on Earth — impacting health, weather, and ecosystems.  
**EnviroScan** provides an automated, end-to-end data pipeline and dashboard for analyzing **global pollution and weather trends**.  

This system:
- Fetches **real-time air quality and meteorological data** across multiple continents.
- Cleans and merges datasets for consistency.
- Trains **machine learning models** (Random Forest, XGBoost, Decision Tree) to classify **pollution sources**.
- Visualizes the results on a **global interactive dashboard**.

🌐 **Live Dashboard:** [https://enviroscan-keerthi.streamlit.app/](https://enviroscan-keerthi.streamlit.app/)

---

## 🚀 Features

✅ Fetches **live air quality** data for multiple countries using [OpenAQ API](https://docs.openaq.org/)  
✅ Fetches **weather parameters** from [OpenWeather API](https://openweathermap.org/api)  
✅ Merges, cleans, and engineers datasets for model training  
✅ Trains **machine learning models** to identify dominant pollution sources  
✅ Provides a **global interactive dashboard** for visualization and exploration  
✅ Generates **high-risk alerts** for polluted regions worldwide  

---

## 🔁 Project Workflow

```mermaid
graph TD
A[Fetch OpenAQ Global Data] --> B[Fetch Pollution Data]
B --> C[Fetch Weather Data]
C --> D[Merge Global Datasets]
D --> E[Feature Engineering]
E --> F[Train ML Models]
F --> G[Streamlit Dashboard Visualization]
````

---

## 🧾 Dataset Description

| Source      | API Used         | Description                                      | Output File                    |
| ----------- | ---------------- | ------------------------------------------------ | ------------------------------ |
| OpenAQ      | `/locations`     | Global monitoring station metadata               | `global_locations_cleaned.csv` |
| OpenWeather | `/air_pollution` | Real-time air quality data                       | `pollution_data.csv`           |
| OpenWeather | `/weather`       | Meteorological parameters (temp, humidity, wind) | `weather_data.csv`             |
| Merged      | -                | Combined dataset with geolocation info           | `merged_core_data.csv`         |
| Cleaned     | -                | Feature-engineered dataset                       | `cleaned_features_new.csv`     |

---

## ⚙️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/enviro-scan.git
cd enviro-scan
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the root directory:

```
OPENAQ_API_KEY=your_openaq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

---

## 📜 Script Descriptions

| Script                                             | Function                                                           | Output                         |
| -------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------ |
| **`scripts/fetch_openaq.py`**                      | Fetches global air quality monitoring locations                    | `global_locations_cleaned.csv` |
| **`scripts/fetch_pollution.py`**                   | Retrieves global pollution readings (PM2.5, PM10, NO₂, etc.)       | `pollution_data.csv`           |
| **`scripts/fetch_weather_retry_missing_fixed.py`** | Collects weather data for each location                            | `weather_data.csv`             |
| **`scripts/merge_new.py`**                         | Combines pollution, weather, and location data                     | `merged_core_data.csv`         |
| **`scripts/dfandfe_new.py`**                       | Performs data cleaning and feature engineering                     | `cleaned_features_new.csv`     |
| **`scripts/model_training_optimized.py`**          | Trains optimized ML models (Random Forest, XGBoost, Decision Tree) | `/models/*.joblib`             |
| **`scripts/streamlit_dashboard.py`**               | Creates interactive dashboard for visualization                    | Streamlit web app              |

---

## ▶️ Running the Pipeline

Execute the following in order:

```bash
python scripts/fetch_openaq.py
python scripts/fetch_pollution.py
python scripts/fetch_weather_retry_missing_fixed.py
python scripts/merge_new.py
python scripts/dfandfe_new.py
python scripts/model_training_optimized.py
```

To launch the Streamlit dashboard locally:

```bash
streamlit run scripts/streamlit_dashboard.py
```

Or access the deployed dashboard here:
🌐 **[EnviroScan Live Dashboard](https://enviroscan-keerthi.streamlit.app/)**

---

## 🤖 Model Training and Outputs

**Models Used:**

* Decision Tree Classifier
* Random Forest Classifier (with `RandomizedSearchCV`)
* XGBoost Classifier (with `GridSearchCV`)

**Performance Metrics:**

* Accuracy Score
* Classification Report
* Confusion Matrix

**Saved Models:**

* `models/random_forest_model.joblib`
* `models/xgb_model.joblib`
* `models/decision_tree_model.joblib`
* `models/scaler.joblib`
* `models/label_encoder.joblib`

---

## 📊 Streamlit Dashboard

**Deployed App:**
🌐 [https://enviroscan-keerthi.streamlit.app/](https://enviroscan-keerthi.streamlit.app/)

**Dashboard Features:**

* 🗺 **Global Map:** Real-time visualization of pollution sources
* 🔥 **Heatmap:** Intensity visualization of pollutants across regions
* 📈 **Trend Analysis:** Average pollutant levels over time
* 🧩 **Source Classification:** Predicted sources (vehicular, industrial, etc.)
* ⚠️ **Alert System:** High-risk zones flagged dynamically

---

## 📁 Directory Structure

```
├── data/
│   ├── global_locations_cleaned.csv
│   ├── pollution_data.csv
│   ├── weather_data.csv
│   ├── merged_core_data.csv
│   └── cleaned_features_new.csv
│
├── models/
│   ├── random_forest_model.joblib
│   ├── xgb_model.joblib
│   ├── decision_tree_model.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
│
├── scripts/
│   ├── fetch_openaq.py
│   ├── fetch_pollution.py
│   ├── fetch_weather_retry_missing_fixed.py
│   ├── merge_new.py
│   ├── dfandfe_new.py
│   ├── model_training_optimized.py
│   └── streamlit_dashboard.py
│
├── .env
├── requirements.txt
├── README.md
└── config.py
```

---

## 🔮 Future Enhancements

* 🌐 Add global time-series forecasting using **Prophet/LSTM**
* 🌎 Integrate satellite-based pollution datasets (e.g., NASA MODIS, Sentinel-5P)
* 🔔 Real-time notifications for air quality threshold breaches
* ☁️ Deploy dashboard with automated daily data refresh

---

## 👩‍💻 Contributor

**Thatikonda Sai Keerthi**
🎓 Integrated M.Tech (CSE - Business Analytics)
📧 [GitHub Profile](https://github.com/<your-username>)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---


