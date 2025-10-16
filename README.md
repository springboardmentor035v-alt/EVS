# EnviroScan: Andhra Pradesh Pollution Monitoring Dashboard

<img src="enviroscan_banner.png" width="70%" alt="EnviroScan Banner" />

**EnviroScan** is an advanced, interactive dashboard for monitoring, analyzing, visualizing, and forecasting air pollution across Andhra Pradesh, India. Built with Python and Streamlit, it empowers users to view real-time, historical, and predictive AQI data, explore pollution sources, and receive SMS alerts for unsafe conditions.

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Credits](#credits)
- [License](#license)

---

## Features

- 📊 **Pollution Trends:** Visualize time series trends for PM2.5, PM10, NO2, SO2, CO, and Ozone for any Andhra Pradesh city.
- 🗺️ **Interactive Map & Alerts:** Explore AQI and pollutant levels on a city map with vivid heatmaps and marker clusters for speed and clarity. Real-time pollution alerts shown for selected city.
- 🌀 **Source Distribution:** Pie chart analysis of pollution sources (Vehicular, Industrial, etc.) by city or filter.
- ⏳ **Historical View:** See how air quality changed in the past, city-wise and pollutant-wise.
- 🕒 **Future Prediction:** Select a future date and get an AQI forecast and category (Good, Satisfactory, Moderate, etc.) for your city and pollutant.
- 📩 **SMS Alerts:** Get SMS notifications for AQI breaches (via Twilio).
- 📑 **Download Data:** Export city-filtered AQI and pollutant data to CSV.
- 💡 **Beautiful UI:** Dark sidebar, attractive navigation, project/about tab, and custom imagery.
- ⚡ **Fast Filtering:** By city, pollutant, and date range (separate start and end date pickers).
- 🖼️ **Custom Images:** Banner image and sidebar logo support.

---

## Getting Started

### Prerequisites

- Python 3.8+
- `streamlit`
- `pandas`
- `folium`
- `streamlit_folium`
- `matplotlib`
- `numpy`
- `twilio`
- `Pillow`

### Installation

Clone the repo and install dependencies:
git clone https://github.com/teja_nareddy/enviroscan.git
cd enviroscan
pip install -r requirements.txt


## Project Structure

├── processed_pollution_data.csv
├── enviroscan_banner.png
├── side_img.webp
├── dashboard.py
├── requirements.txt
├── .streamlit/
│ ├── config.toml
│ └── secrets.toml
├── screenshots/
│ ├── dashboard_main.png
│ ├── dashboard_map.png
│ └── source_distribution.png

---

## Configuration

Set the following [Streamlit secret variables](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management) for Twilio SMS alerts:

.streamlit/secrets.toml
twilio_account_sid = "your_twilio_account_sid"
twilio_auth_token = "your_twilio_auth_token"
twilio_from_number = "+1234567890"

---

## Usage

1. **Run the app:**

streamlit run dashboard.py

text

2. **Choose a city**, select date range (start/end), pick a pollutant, and explore dashboard sections using the four navigation buttons:
- Pollution Trends
- Source Distribution
- Map & Alerts (shows heatmap and popups for all pollutants in city)
- Future Prediction (select a future date & pollutant, click "Predict AQI")

3. **Optionally enable SMS alerts** or download filtered data to CSV.

---

## Tech Stack

- **Python** 
- **Streamlit** (_dashboard app, widgets, styling_)
- **Pandas** (_data handling_)
- **Folium & streamlit_folium** (_responsive maps, heatmaps, clusters_)
- **Matplotlib** (_data visualization_)
- **NumPy** (_calculation, AQI calculation_)
- **Twilio** (_SMS alerts_)
- **Pillow** (_image processing for side/banner images_)

---

## Credits

- Datasets: [OpenAQ](https://openaq.org/) and Andhra Pradesh Pollution Board, plus custom processed sources.
- Banner/sidebar images: (Google images).

---

## License

[MIT](LICENSE)

---

_For questions or collaboration, please open an issue or contact tejanareddy06@gmail.com
