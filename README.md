# EnviroScan-AI-Powered-Team-6 

# Website URL (Deployment link):

https://enviroscanapp.streamlit.app/


https://github.com/user-attachments/assets/eb34593d-9348-4055-ada0-b9fb68e0233e


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

