import os
import joblib
import hopsworks
import pandas as pd
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import pytz # Add this to the top of your app.py imports

load_dotenv()
api_key_weather = os.getenv("WeatherAPI_KEY")
api_key_hopswork = os.getenv("HopsworkAPI_KEY")

st.set_page_config(page_title="Sialkot AQI Forecaster", page_icon="🌤️")
st.title("🌤️ Sialkot Air Quality Index (AQI) 3-Day Forecast")
st.write("A fully automated, real-time air quality platform that ingests live environmental data every hour and leverages cloud-based machine learning to predict localized conditions 72 hours in advance.")

# 1. Load the model in cache
@st.cache_resource
def get_model():
    load_dotenv()
    project = hopsworks.login(api_key_value=api_key_hopswork)
    mr = project.get_model_registry()
    
    # Get the model from uploaded hopswork
    model_meta = mr.get_model("sialkot_pm2_5_predictor_v2")
    model_dir = model_meta.download()
    
    # Load the actual .pkl file
    model = joblib.load(model_dir + "/sialkot_pm25_model.pkl")
    return model

# 2. Fetch the data
# Function to get forcast pollution data via openMeteo
def get_weather_forcast(base_url_history_weather):
    response_weather = requests.get(url=base_url_history_weather)
    weather = response_weather.json()
    weather = weather["hourly"]
    weather_data = pd.DataFrame(weather)
    weather_data.rename(columns={'time':'datetime'}, inplace=True)
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    
    return weather_data

# Function to get forcast pollution data via openWeather
def get_pollution_forcast(base_url_history_pollution):
    response_pollution = requests.get(url=base_url_history_pollution)
    pollution = response_pollution.json()

    temp = []

    pollution = pollution['list']

    for hourly_data in pollution:
        sample = {}
        sample["aqi"] = hourly_data['main']["aqi"]
        sample['co'] = hourly_data['components']['co']
        sample['no2'] = hourly_data['components']['no2']
        sample['o3'] = hourly_data['components']['o3']
        sample['so2'] = hourly_data['components']['so2']
        sample['pm2_5'] = hourly_data['components']['pm2_5']
        sample['pm10'] = hourly_data['components']['pm10']
        sample['nh3'] = hourly_data['components']['nh3']
        sample["dt"] = hourly_data['dt']
        temp.append(sample)

    pollution_data = pd.DataFrame(temp)
    pollution_data['datetime'] = pd.to_datetime(pollution_data['dt'], unit='s')
    pollution_data['date'] = pollution_data['datetime'].dt.date
    pollution_data['hour'] = pollution_data['datetime'].dt.hour
    pollution_data = pollution_data.drop('dt',axis=1)

    return pollution_data

@st.cache_data(ttl=3600) # Caches the weather data for 1 hour
def fetch_weather_forecast():
    """Fetches the 3-day weather forecast for Sialkot."""
    latitude = 32.49
    longitude = 74.54
    
    # Should ask for the next 3 days of hourly data for the exact same features you trained on:
    # wind_speed_100m, precipitation, apparent_temperature, wind_gusts_10m, vapour_pressure_deficit, relative_humidity_2m
    # Dynamic Time Calculation
    # Pull 4 days of data to ensure no hours are missed between runs

    # Using OpenMeteo
    forecast_weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=wind_speed_100m,precipitation,apparent_temperature,wind_gusts_10m,vapour_pressure_deficit,relative_humidity_2m&timezone=GMT&forecast_days=5&past_days=1"

    # Using OpenWeather 
    forcast_pollutants_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={latitude}&lon={longitude}&appid={api_key_weather}"

    forecast_weather_df = get_weather_forcast(forecast_weather_url)
    forecast_pollutants_df = get_pollution_forcast(forcast_pollutants_url)
    forecast_df = pd.merge(forecast_weather_df,forecast_pollutants_df, on='datetime', how='left')

    return forecast_df

# 3. Conversion logic to US AQI scale
def calculate_us_aqi(pm25):
    # Breakpoints: (C_low, C_high, AQI_low, AQI_high)
    breakpoints = [
        (0.0, 12.0, 0, 50),       # Good
        (12.1, 35.4, 51, 100),    # Moderate
        (35.5, 55.4, 101, 150),   # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),  # Unhealthy
        (150.5, 250.4, 201, 300), # Very Unhealthy
        (250.5, 350.4, 301, 400), # Hazardous
        (350.5, 500.4, 401, 500)  # Hazardous (Beyond Index)
    ]
    
    # EPA requires rounding PM2.5 to 1 decimal place before calculating
    c = round(pm25, 1)
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return int(round(aqi))
            
    # If Sialkot smog goes literally off the charts
    if c > 500.4:
        return 500 
    
    return 0

# 4. Translation the aqi value for simple context function
def get_aqi_context(aqi_value):
    """Returns the human label, emoji, and advice based on US AQI."""
    if aqi_value <= 50:
        return {"label": "Good", "emoji": "🌲", "color": "green", "advice": "Air is clean. Perfect day for a run!"}
    elif aqi_value <= 100:
        return {"label": "Moderate", "emoji": "🙂", "color": "yellow", "advice": "Acceptable air quality for most people."}
    elif aqi_value <= 150:
        return {"label": "Sensitive", "emoji": "😷", "color": "orange", "advice": "Sensitive groups should reduce heavy outdoor exertion."}
    elif aqi_value <= 200:
        return {"label": "Unhealthy", "emoji": "⚠️", "color": "red", "advice": "Unhealthy. Wear a mask outdoors."}
    elif aqi_value <= 300:
        return {"label": "Very Unhealthy", "emoji": "🛑", "color": "purple", "advice": "Dangerous air quality. Limit all outdoor activities."}
    else:
        return {"label": "Hazardous", "emoji": "☠️", "color": "maroon", "advice": "Health warning of emergency conditions. Stay indoors."}

# 5. Execute and Display
with st.spinner("Downloading AI Model and Live Weather..."):
    rf_model = get_model()
    future_weather_df = fetch_weather_forecast()

future_weather_df['temp_lag1'] = future_weather_df['apparent_temperature'].shift(1)
future_weather_df['wind_rolling_6'] = future_weather_df['wind_speed_100m'].rolling(6).mean()

future_weather_df['day_of_week'] = future_weather_df['datetime'].dt.dayofweek
future_weather_df['month']       = future_weather_df['datetime'].dt.month
future_weather_df['is_weekend']  = future_weather_df['day_of_week'].isin([5, 6]).astype(int)

# Convert the 24-hour clock into a mathematical circle using Sine and Cosine
future_weather_df['hour_sin'] = np.sin(2 * np.pi * future_weather_df['hour'] / 24)
future_weather_df['hour_cos'] = np.cos(2 * np.pi * future_weather_df['hour'] / 24)

future_weather_df = future_weather_df.ffill()

future_weather_df = future_weather_df.dropna().reset_index(drop=True)

# Prepare the exact columns the model expects and used in training
features_for_prediction = future_weather_df[['wind_speed_100m', 'precipitation', 'apparent_temperature', 
                                             'wind_gusts_10m', 'vapour_pressure_deficit', 
                                             'relative_humidity_2m', 'co', 'no2','o3','so2','nh3','temp_lag1',
                                             'wind_rolling_6','day_of_week','month',
                                             'is_weekend','hour_sin','hour_cos']]

# Make the PM2.5 Predictions
future_weather_df['Predicted_PM25'] = rf_model.predict(features_for_prediction)

# Convert to AQI Bucket
future_weather_df['Predicted_AQI'] = future_weather_df['Predicted_PM25'].apply(calculate_us_aqi)

# Dashboard View
st.subheader("72-Hour PM2.5 Forecast")
st.line_chart(data=future_weather_df, x='datetime', y='Predicted_PM25')

st.markdown("---")
st.subheader("Daily AQI Forecast (Next 3 Days)")

# Extract the raw date (removes the hours)
future_weather_df['date'] = future_weather_df['datetime'].dt.date

# Filter out "Today"
pkt_tz = pytz.timezone('Asia/Karachi')
today = datetime.now(pkt_tz).date()
future_weather_df = future_weather_df[future_weather_df['date'] > today]

# Ensure we ONLY have exactly the next 3 days (OpenWeather might return 4 or 5)
next_3_days = sorted(future_weather_df['date'].unique())[:3]
future_weather_df = future_weather_df[future_weather_df['date'].isin(next_3_days)]

# Group by Date and calculate the Average PM2.5
daily_summary_df = future_weather_df.groupby('date')['Predicted_PM25'].mean().reset_index()

# Convert the Daily Average PM2.5 into our 1-5 AQI Category
daily_summary_df['Predicted_AQI'] = daily_summary_df['Predicted_PM25'].apply(calculate_us_aqi)

# Alert System
# Find the absolute worst AQI predicted in the next 3 days
worst_forecast_aqi = daily_summary_df['Predicted_AQI'].max()

if worst_forecast_aqi >= 151: # Unhealthy or worse
    st.error(f"**AIR QUALITY ALERT:** Hazardous smog conditions expected this week. Peak AQI will reach **{worst_forecast_aqi}**. Wear an N95 mask outdoors.")
elif worst_forecast_aqi >= 101: # Sensitive groups
    st.warning(f"**AIR QUALITY ADVISORY:** Smog levels are rising. Peak AQI expected to reach **{worst_forecast_aqi}**. Sensitive individuals should take precautions.")
else:
    st.success(f"**ALL CLEAR:** Sialkot air quality looks great for the next 3 days! Peak AQI: **{worst_forecast_aqi}**.")
st.markdown("---")

# Streamlit UI: Create 3 side-by-side metric cards
cols = st.columns(3)

# Loop through our 3 days and populate the UI columns
for index, row in daily_summary_df.iterrows():
    display_date = row['date'].strftime('%b %d')
    
    daily_aqi = int(row['Predicted_AQI'])
    daily_pm25 = round(row['Predicted_PM25'], 1)
    
    # Grab the human context for this specific AQI number
    context = get_aqi_context(daily_aqi)
    
    with cols[index]:
        # the metric card with the label and emoji
        st.metric(
            label=f"{display_date} | {context['label']} {context['emoji']}", 
            value=f"AQI: {daily_aqi}", 
            delta=f"{daily_pm25} µg/m³ PM2.5", 
            delta_color="off" 
        )
        # Add the actionable health advice right below the card
        st.caption(f"**Advice:** {context['advice']}")

st.markdown("---")
st.write("### Raw Forecast Data (Hourly)")

# Hourly prediction
st.dataframe(future_weather_df[['datetime', 'Predicted_PM25', 'Predicted_AQI']])
