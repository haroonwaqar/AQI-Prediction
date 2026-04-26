import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # This searches for a .env file and loads it
api_key = os.getenv("WeatherAPI_KEY")
latitude = 32.49 # sialkot
longitude = 74.54 # sialkot
start = 1775001600
end = 1775088000


# main.aqi
# Air Quality Index. 
# Possible values: 1, 2, 3, 4, 5. 
# Where 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor.

#base_url_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid={api_key}"
#base_url_weather = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&units=metric&appid={api_key}"
base_url_history_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={latitude}&lon={longitude}&start={start}&end={end}&appid={api_key}"

#features = ['main', 'wind', 'rain', 'visibility']

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

data = pd.DataFrame(temp)
data['datetime'] = pd.to_datetime(data['dt'], unit='s')
data['date'] = data['datetime'].dt.date
data['hour'] = data['datetime'].dt.hour
data = data.drop('dt',axis=1)

print(data.head())





