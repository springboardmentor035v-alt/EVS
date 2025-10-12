🌍 EnviroScan: Real-Time Environmental Monitoring and Forecasting Dashboard

🧠 Overview

EnviroScan is an interactive Streamlit web application designed to visualize, analyze, and forecast environmental pollution levels across Indian cities.
It combines real-time pollution data visualization, statistical insights, and predictive modeling to support smarter urban planning and public health awareness.

🚀 Features

🌆 City-Based Pollution Insights
Type and search for any city to view detailed air quality and weather metrics.
Displays pollutants such as PM2.5, PM10, NO₂, SO₂, CO, and O₃.

📊 Interactive Visualization

Bar chart comparison of multiple pollutants for selected cities

Real-time pollution map using latitude and longitude

Circle markers representing pollutant intensity


⚠ Alert System
Automatically flags cities with poor air quality or unsafe pollution levels.

📈 Forecasting Module (NEW)
Predicts the next 7 days of Air Quality Index (AQI) using Linear Regression based on historical trends.
Future upgrade includes LSTM (Long Short-Term Memory) deep learning model for advanced forecasting.

💡 City Analytics Table
Displays only the searched city’s data (not the entire dataset) — including weather conditions, temperature, humidity, and infrastructure indices.


🧰 Tech Stack

Category	Tools / Libraries Used

Frontend	Streamlit
Backend / Logic	Python
Data Handling	Pandas, NumPy
Visualization	Plotly, Matplotlib
Machine Learning	Scikit-learn (Linear Regression), (Future: TensorFlow LSTM)
Map Integration	Streamlit’s built-in st.map() and pydeck
Version Control	Git & GitHub
Deployment	Streamlit Cloud


📂 Project Structure

enviroScanProject/
│
├── collected_data_cleaned.csv      # Dataset containing city pollution and weather data
├── streamlit_app.py                # Main Streamlit app
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── images/                         # (optional) Screenshots for GitHub preview


🧾 Dataset Description

The dataset (collected_data_cleaned.csv) contains:

City

Latitude / Longitude

Pollutant levels: PM2.5, PM10, NO₂, SO₂, CO, O₃

Weather Data: Temperature, Humidity, Weather Condition

Infrastructure Data: Roads Count, Factories Count, Infra Score

Derived Features: Temperature Z-score, Humidity Z-score, Pollution Count


🌐 Deployment

The project is deployed on Streamlit Cloud.
Visit:
🔗 https://enviroscanproject-xrzxbbqkxusymnzkr5ornx.streamlit.app/



🧮 Forecasting Logic

Model: Linear Regression (sklearn.linear_model)

Input: Historical AQI data of the selected city

Output: Predicted AQI for the next 7 days

Visualization: Line chart with actual vs predicted AQI


> 🧩 Future Plan: Implement LSTM (Long Short-Term Memory) Neural Network for more accurate time-series forecasting.



💭 Future Enhancements

Integrate LSTM model for more accurate multi-feature forecasting (pollutants + weather)

Add downloadable city-wise PDF reports

Include real-time API data (from OpenWeatherMap / CPCB)

Build mobile-responsive version of the dashboard


👩‍💻 Author

Vaishnavi P


🪄 Acknowledgements

Special thanks to my mentor for guidance and continuous feedback in enhancing this project.
Thanks to the Streamlit and Scikit-learn communities for open-source support.
