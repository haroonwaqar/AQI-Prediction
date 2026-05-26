# file to use to get pollution history
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # This searches for a .env file and loads it
api_key = os.getenv("WeatherAPI_KEY")

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







