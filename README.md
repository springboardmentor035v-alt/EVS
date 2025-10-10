# 🏭 Pollution Source Identifier using AI and Geospatial Analytics

## Project Summary
This project aims to classify the likely source of pollution (Industrial, Vehicular, Natural) using machine learning on air quality and environmental parameters, and visualize the results on a real-time, interactive geospatial dashboard.

---

## 🚀 Status: Minimum Viable Product (MVP) 

### Modules Completed:
* **Module 3 & 4 (Source Labeling and ML Model):** Implemented a multi-class Random Forest Classifier. The dataset was simulated with labeled sources to create a balanced dataset for training and prediction.
* **Module 5 (Geospatial Mapping):** A Folium-based map is embedded, showing dynamic markers and a Heatmap overlay based on the predicted pollution level.
* **Module 6 (Real-Time Dashboard):** A Streamlit dashboard is fully functional, including filters, real-time alerts (based on a mock threshold of 150), a Pie Chart of source distribution, and a Download Report option.

### ML Model Evaluation Metrics (From `src/03_model_training.py`):
These metrics were achieved on the test set derived from the mock data:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Industrial | 1.00 | 1.00 | 1.00 | 4 |
| Natural | 1.00 | 1.00 | 1.00 | 3 |
| Vehicular | 1.00 | 1.00 | 1.00 | 3 |
| **Overall Accuracy** | **1.00** | | | **10** |

---

## 🛠️ Installation and Execution Guide

### 1. Setup

```bash
# 1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd Pollution-Source-Identifier-Streamlit

# 2. Install dependencies
pip install -r requirements.txt

### 2. Run the Machine Learning Pipeline (Modules 3 & 4)

This command trains the model (generating the metrics above), saves the model file, and updates the data with the final source predictions used by the dashboard.

```bash
python src/03_model_training.py

### 3. Launch the Dashboard (Modules 5 & 6)

Execute the Streamlit application to launch the interactive, real-time dashboard in your web browser.

```bash
streamlit run app.py
