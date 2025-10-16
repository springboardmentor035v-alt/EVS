
# 🌍 EnviroScan AI 

EnviroScan AI is a proof-of-concept platform that demonstrates the power of machine learning in environmental science. It provides actionable insights into the sources of air pollution by analyzing real-time air quality data, weather patterns, and geospatial information.

## 🚀 Live Demo
Check out the deployed app here:  
[EnviroScan AI Web App](https://enviroscanai-kche9zywaakhvqwrcgcezc.streamlit.app/)

https://github.com/user-attachments/assets/4851c79d-62fb-4e3b-a90c-d88e19663952











## 📖 Project Overview
EnviroScan AI aims to go beyond measuring pollution levels by attributing pollution events to specific sources such as:
- Vehicular traffic
- Industrial activity
- Agricultural burning  

**Target Users:**
- Urban Planners & Policymakers  
- Environmental Agencies  
- Researchers & Students  

---

## 🔬 Methodology

### 1. Data Pipeline
- **Ingestion:** Collect air quality metrics (PM₂.₅, NO₂, SO₂), meteorological data, and geospatial features.  
- **Preprocessing:** Clean data, handle missing values, and create time-based features.  
- **Labeling:** Rule-based engine assigns preliminary `pollution_source` labels for supervised learning.  
- **Train/Test Split:** Data is split into training and testing sets for evaluation.  

### 2. Model Training & Evaluation
- **Algorithm:** Ensemble Voting Classifier combining:
  - **Random Forest Classifier (RF)**
  - **XGBoost Classifier (XGB)**
- **Why Ensemble Voting?**
  - Combines strengths of multiple models for higher accuracy
  - Reduces overfitting compared to single models
  - Handles non-linear relationships and complex feature interactions
  - Supports multi-class evaluation with classification metrics

**Evaluation Metrics:**
- Classification report (precision, recall, F1-score for each pollution source class)  
- Confusion matrix (visual analysis of predictions)  

**Model Output:**
- Ensemble model saved as `outputs/ensemble_model.joblib`  
- Label encoder saved as `config.ENCODER_FILE`  
- Evaluation report saved as `outputs/ensemble_model_report.txt`
 

---

## 💻 Technology Stack
- **Python:** Core backend language  
- **Streamlit:** Web app and dashboard  
- **Scikit-learn:** Machine learning models  
- **Pandas:** Data processing and analysis  
- **Folium & Streamlit-Folium:** Interactive geospatial maps  
- **Plotly & Seaborn:** Charts and visualizations  
- **Joblib:** Model serialization  
- **Python-dotenv:** Managing API keys  

---

## ⚙️ How to Run This Project

### Windows

```bash
# 1. Clone repository
git clone https://github.com/Supriyo760/enviroscan.Ai
cd EnviroScan-Project

# 2. Setup environment
=======
How to Run this Project:- 
Step by Step commands :-
---------------------------------
Windows:-
:: 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

:: 2. Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Add API key
notepad .env
# Add this line:
OPENWEATHER_API_KEY=your_key_here

# 4. Run data pipeline
python run_pipeline.py

# 5. Start Streamlit app
streamlit run app.py
````

### MacOS / Linux

```bash
# 1. Clone repository
git clone https://github.com/Supriyo760/enviroscan.Ai
cd EnviroScan-Project
=======
:: 3. Add API key
notepad .env
:: Add this line: OPENWEATHER_API_KEY=your_key_here

:: 4. Run pipeline
python3 run_pipeline.py

:: 5. Start app
streamlit run app.py
-------------------------------------------------------------------------
 Mac os users:-
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Add API key
echo "OPENWEATHER_API_KEY=your_key_here" > .env

<<<<<<< HEAD
# 4. Run data pipeline
python3 run_pipeline.py

# 5. Start Streamlit app
streamlit run app.py
```

### Troubleshooting Dependencies

If some packages fail to install, try installing them individually:

```bash
pip install streamlit pandas scikit-learn xgboost
pip install folium streamlit-folium plotly
pip install requests osmnx python-dotenv joblib
```

---

## 📂 Repository Structure

```
EnviroScan-AI-Powered-Team-6/
│
├── app.py                       # Main Streamlit app entry point
├── run_pipeline.py              # Script to run the entire data pipeline
├── EnviroScan.py                # Possibly main project logic / module
├── train_model.py               # Script for training the ML model
├── config.py                    # Configuration file for constants, paths, etc.
├── requirements.txt             # Project dependencies
├── .env                         # Environment variables (API keys)
├── enviroscan.log               # Log file for the project
├── How_to_run_program_file       # Instructions / script guide
├── my_script_code.txt           # Any additional code snippets
├── README.md                    # Project documentation
│
├── pages/                       # Streamlit multi-page app scripts
│   ├── About.py
│   ├── Live_Prediction.py
│   ├── Pollution_Dashboard.py
│   └── image.png                # Image used in About page
│
├── scripts/                     # Scripts for data pipeline and ML preprocessing
│   ├── 01_data_collection.py
│   ├── 02_data_processing.py
│   ├── 03_label_and_split.py
│   ├── 04_train_model.py
│   └── __pycache__/             # Python cache folder
│
├── venv/                        # Python virtual environment
│
├── data/ (optional, if exists)  # Raw / processed datasets
│   └── locations.csv
│
├── outputs/                     # Any output files, models, or results
│
└── cache/                       # Streamlit or project cache files

```

---

## 📝 License

This project is open-source. Feel free to use, modify, and share under the [MIT License](LICENSE).

---

=======
# 4. Run pipeline
python3 scripts/run_pipeline.py

# 5. Start app
streamlit run app.py
-----------------------------------------------------------------------------
others:-
    
If dependencies fail to install:
# Try installing individually
pip install streamlit pandas scikit-learn xgboost
pip install folium streamlit-folium plotly
pip install requests osmnx python-dotenv joblib


