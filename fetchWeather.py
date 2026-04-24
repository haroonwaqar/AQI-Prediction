import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # This searches for a .env file and loads it
api_key = os.getenv("WeatherAPI_KEY")
latitude = 32.49 # sialkot
longitude = 74.54 # sialkot
start = 1640995200
end = 1776643200

# main.aqi
# Air Quality Index. 
# Possible values: 1, 2, 3, 4, 5. 
# Where 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor.

base_url_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid={api_key}"
base_url_weather = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&units=metric&appid={api_key}"
base_url_history_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={latitude}&lon={longitude}&start={start}&end={end}&appid={api_key}"

features = ['main', 'wind', 'rain', 'visibility']

response_pollution = requests.get(url=base_url_pollution)
response_weather = requests.get(url=base_url_weather)

pollution = response_pollution.json()
pollution = pollution['list']
#weather = response_weather.json()
#weather = weather[features]

print(pollution)
#print(weather)





