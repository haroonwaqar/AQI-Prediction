import os
import joblib
import hopsworks
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
api_key_weather = os.getenv("WeatherAPI_KEY")
api_key_hopswork = os.getenv("HopsworkAPI_KEY")

st.set_page_config(page_title="Sialkot AQI Forecaster", page_icon="🌤️", layout="centered")
st.title("🌤️ Sialkot Air Quality Index (AQI) 3-Day Forecast")
st.write("developed using OpenWeather, OpenMeteo, and Hopsworks")

# 1. Load the model in cache
@st.cache_resource
def get_model():
    load_dotenv()
    project = hopsworks.login(api_key_value=api_key_hopswork)
    mr = project.get_model_registry()
    
    # Get the model from uploaded hopswork
    model_meta = mr.get_model("sialkot_pm2_5_predictor")
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
    forecast_weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=wind_speed_100m,precipitation,apparent_temperature,wind_gusts_10m,vapour_pressure_deficit,relative_humidity_2m&timezone=GMT&forecast_days=4"

    # Using OpenWeather 
    forcast_pollutants_url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={latitude}&lon={longitude}&appid={api_key_weather}"

    forecast_weather_df = get_weather_forcast(forecast_weather_url)
    forecast_pollutants_df = get_pollution_forcast(forcast_pollutants_url)
    forecast_df = pd.merge(forecast_weather_df,forecast_pollutants_df,"inner","datetime")
    
    return forecast_df

# 3. Conversion logic
def convert_pm25_to_aqi_category(pm25_value):
    # Converts continuous PM2.5 into the categorical 1-5 AQI scale.
    if pm25_value > 0 and pm25_value < 10:
        aqi = 1
    elif pm25_value >= 10 and pm25_value < 25:
        aqi = 2
    elif pm25_value >= 25 and pm25_value < 50:
        aqi = 3
    elif pm25_value >= 50 and pm25_value < 75:
        aqi = 4
    elif pm25_value >= 75:
        aqi = 5
    else:
        aqi = 0
    
    return aqi

# Translation dictionary
AQI_CONTEXT = {
    1: {"label": "Good", "emoji": "🌲", "advice": "Air is clean. Perfect day for a run!"},
    2: {"label": "Fair", "emoji": "🙂", "advice": "Air quality is acceptable for most people."},
    3: {"label": "Moderate", "emoji": "😷", "advice": "Sensitive groups should reduce heavy outdoor exertion."},
    4: {"label": "Poor", "emoji": "⚠️", "advice": "Unhealthy. Wear a mask outdoors and keep windows closed."},
    5: {"label": "Hazardous", "emoji": "☠️", "advice": "Dangerous air quality. Stay indoors and use purifiers."}
}

# 4. Execute and Display
with st.spinner("Downloading AI Model and Live Weather..."):
    rf_model = get_model()
    future_weather_df = fetch_weather_forecast()

# Prepare the exact columns the model expects and used in training
features_for_prediction = future_weather_df[['wind_speed_100m', 'precipitation', 'apparent_temperature', 
                                             'wind_gusts_10m', 'vapour_pressure_deficit', 
                                             'relative_humidity_2m', 'co', 'no2','o3','so2','nh3','hour']]

# Make the PM2.5 Predictions
future_weather_df['Predicted_PM25'] = rf_model.predict(features_for_prediction)

# Convert to AQI Bucket
future_weather_df['Predicted_AQI_Level'] = future_weather_df['Predicted_PM25'].apply(convert_pm25_to_aqi_category)

# Dashboard View
st.subheader("72-Hour PM2.5 Forecast")
st.line_chart(data=future_weather_df, x='datetime', y='Predicted_PM25')

st.markdown("---")
st.subheader("Daily AQI Forecast (Next 3 Days)")

# Extract the raw date (removes the hours)
future_weather_df['date'] = future_weather_df['datetime'].dt.date

# Filter out "Today"
today = datetime.now().date()
future_weather_df = future_weather_df[future_weather_df['date'] > today]

# Ensure we ONLY have exactly the next 3 days (OpenWeather might return 4 or 5)
next_3_days = sorted(future_weather_df['date'].unique())[:3]
future_weather_df = future_weather_df[future_weather_df['date'].isin(next_3_days)]

# Group by Date and calculate the Average PM2.5
daily_summary_df = future_weather_df.groupby('date')['Predicted_PM25'].mean().reset_index()

# Convert the Daily Average PM2.5 into our 1-5 AQI Category
daily_summary_df['Predicted_AQI_Level'] = daily_summary_df['Predicted_PM25'].apply(convert_pm25_to_aqi_category)

# Streamlit UI: Create 3 side-by-side metric cards
cols = st.columns(3)

# Loop through our 3 days and populate the UI columns
for index, row in daily_summary_df.iterrows():
    display_date = row['date'].strftime('%b %d')
    
    daily_aqi = int(row['Predicted_AQI_Level'])
    daily_pm25 = round(row['Predicted_PM25'], 1)
    
    # Grab the human context for this specific AQI number
    context = AQI_CONTEXT.get(daily_aqi, {"label": "Unknown", "emoji": "❓", "advice": "Data unavailable."})
    
    with cols[index]:
        # the metric card with the label and emoji
        st.metric(
            label=f"{display_date} | {context['label']} {context['emoji']}", 
            value=f"Level {daily_aqi}", 
            delta=f"{daily_pm25} µg/m³ PM2.5", 
            delta_color="off" 
        )
        
        # Add the actionable health advice right below the card
        st.caption(f"**Advice:** {context['advice']}")

st.markdown("---")
st.write("### Raw Forecast Data (Hourly)")
# Hourly prediction
st.dataframe(future_weather_df[['datetime', 'Predicted_PM25', 'Predicted_AQI_Level']])
