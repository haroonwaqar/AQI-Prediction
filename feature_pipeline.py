import os
import requests
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from datetime import datetime, timedelta

latitude = 32.49 # sialkot
longitude = 74.54 # sialkot

# Use this to run first time so this time frame data is uploaded on Hopswork. Then shift to Dynamic so GitHub Actions can Work.
# start = 1772204400 # 27 Feb 2026
# end = 1777302000 # 27 April 2026

# start_date = datetime.fromtimestamp(start)
# end_date = datetime.fromtimestamp(end)
# start_date = start_date.date()
# end_date = end_date.date()

# Function to get historical pollution data via openWeather
def get_weather_history(base_url_history_weather):
    response_weather = requests.get(url=base_url_history_weather)
    weather = response_weather.json()
    weather = weather["hourly"]
    weather_data = pd.DataFrame(weather)
    weather_data.rename(columns={'time':'datetime'}, inplace=True)
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    
    return weather_data

# Function to get historical pollution data via openWeather
def get_pollution_history(base_url_history_pollution):
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

def push_to_feature_store(df,api_key):
    # Hopsworks login and fg.insert()
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    aqi_fg = fs.get_or_create_feature_group(
        name="sialkot_aqi_features",
        version=1,
        primary_key=["city"],         # The 'thing' we are measuring
        event_time="datetime",        # 'When' we measured it
        description="Hourly AQI and Open-Meteo weather features for Sialkot"
    )

    # 4. Insert the data!
    aqi_fg.insert(df)


if __name__ == '__main__':

    print("Start Pipline")
    load_dotenv()  # This searches for a .env file and loads it
    api_key_weather = os.getenv("WeatherAPI_KEY")
    api_key_hopswork = os.getenv("HopsworkAPI_KEY")

    # # Using OpenWeather 
    # base_url_history_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={latitude}&lon={longitude}&start={start}&end={end}&appid={api_key_weather}"
    # # Using open meteo 
    # base_url_history_weather = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&hourly=wind_speed_100m,precipitation,apparent_temperature,wind_gusts_10m,vapour_pressure_deficit,relative_humidity_2m&timezone=GMT"
    # #features = ['main', 'wind', 'rain', 'visibility']
    
    # Dynamic Time Calculation
    # We pull 2 days of data to ensure no hours are missed between runs
    now = datetime.now()
    two_days_ago = now - timedelta(days=2)

    # OpenWeather requires UNIX timestamps (integers)
    start_unix = int(two_days_ago.timestamp())
    end_unix = int(now.timestamp())

    # Open-Meteo requires string formats (YYYY-MM-DD)
    start_str = two_days_ago.strftime('%Y-%m-%d')
    end_str = now.strftime('%Y-%m-%d')

    print(f"Fetching data from {start_str} to {end_str}")
    
    # Dynamic Urls
    # Using OpenWeather 
    base_url_history_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={latitude}&lon={longitude}&start={start_unix}&end={end_unix}&appid={api_key_weather}"
    # Using open meteo 
    base_url_history_weather = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={start_str}&end_date={end_str}&hourly=wind_speed_100m,precipitation,apparent_temperature,wind_gusts_10m,vapour_pressure_deficit,relative_humidity_2m&timezone=GMT"

    print("Fetching from APIs")   
    pollution_data_df = get_pollution_history(base_url_history_pollution=base_url_history_pollution)
    weather_data_df = get_weather_history(base_url_history_weather=base_url_history_weather)
    
    print("Merging the Data")
    final_df = pd.merge(weather_data_df,pollution_data_df,"inner","datetime")
    final_df["city"] = "Sialkot"

    print(f"Prepared {len(final_df)} rows for upload")

    print("Pushing to Hopsworks")
    push_to_feature_store(final_df,api_key_hopswork)

    print("Ending Pipline")