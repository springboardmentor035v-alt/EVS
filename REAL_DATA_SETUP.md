# Real AQI Data Setup Guide

## 🌟 Your dashboard now supports REAL air quality data!

### 📊 Data Sources Available:

1. **World Air Quality Index (WAQI)** - Free API with good Indian city coverage
2. **OpenWeather Air Pollution API** - Comprehensive data with coordinates
3. **Fallback Synthetic Data** - Demo data when APIs unavailable

---

## 🔧 Setup Instructions:

### Step 1: Get Free API Keys

#### WAQI API (Recommended - Free):
1. Visit [aqicn.org/api/](https://aqicn.org/api/)
2. Create free account
3. Get your API token
4. Update `real_aqi_data.py` line 14: `self.waqi_token = "YOUR_TOKEN_HERE"`

#### OpenWeather API (Optional):
1. Visit [openweathermap.org/api](https://openweathermap.org/api)
2. Sign up for free account
3. Get API key
4. Update `real_aqi_data.py` line 13: `self.openweather_api_key = "YOUR_KEY_HERE"`

### Step 2: Configure APIs

Edit `real_aqi_data.py`:
```python
# Replace these lines with your actual API keys
self.openweather_api_key = "your_openweather_key_here"
self.waqi_token = "your_waqi_token_here"
```

### Step 3: Test Connection

1. Run your dashboard: `streamlit run streamlit_dashboard.py`
2. In sidebar, check "Use Real AQI Data"
3. Click "Test API Connection"
4. Verify successful connection

---

## 📱 Features with Real Data:

### ✅ What Works Now:
- **Real-time AQI** from monitoring stations
- **Live pollutant levels** (PM2.5, NO2, CO, etc.)
- **Station-specific data** for multiple locations
- **Automatic fallback** to demo data if APIs fail
- **Data source indicators** in sidebar
- **Timestamp tracking** for last update

### 🚀 Enhanced Features:
- **Historical data** based on recent real readings
- **City-specific variations** using actual baselines
- **Time-based updates** with caching for performance
- **Error handling** with graceful degradation

---

## 🎯 Usage Tips:

1. **Start with WAQI API** - It's free and reliable
2. **Check sidebar status** - Shows data source and update time
3. **Enable real data toggle** - Switch between real and demo data
4. **Test connections** - Use the test button to verify APIs
5. **Cache refreshes** - Real data updates every 5 minutes

---

## 🔍 Troubleshooting:

### If APIs don't work:
- Check your internet connection
- Verify API keys are correct
- Ensure you haven't exceeded API limits
- Dashboard will automatically use demo data as fallback

### If data seems wrong:
- Some stations may have temporary outages
- API data quality varies by location
- Cross-reference with official pollution monitoring sites

---

## 🌟 Benefits of Real Data:

- **Accurate current conditions** for better health decisions
- **Real pollution patterns** reflecting actual city conditions
- **Live updates** throughout the day
- **Professional credibility** for presentations
- **Educational value** showing real environmental data

---

## 📋 Next Steps:

1. **Get your free WAQI API key** (5 minutes)
2. **Update the configuration** in `real_aqi_data.py`
3. **Enable real data** in the dashboard sidebar
4. **Test with different cities** to see live variations
5. **Share your dashboard** with real environmental data!

Your EnviroScan dashboard is now capable of displaying real-world air quality data! 🌱📊