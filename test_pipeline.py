import os
import requests
import pandas as pd
from dotenv import load_dotenv
from fetchPollution import get_pollution_history
from fetchWeather import get_weather_history
import hopsworks

load_dotenv()  # This searches for a .env file and loads it
api_key = os.getenv("WeatherAPI_KEY")
latitude = 32.49 # sialkot
longitude = 74.54 # sialkot
#start = 1775001600 # 1 April 2026
#end = 1775602800 # 7 April 2026
start = 1772204400 # 27 Feb 2026
end = 1777302000 # 27 April 2026

# main.aqi
# Air Quality Index. 
# Possible values: 1, 2, 3, 4, 5. 
# Where 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor.

base_url_history_pollution = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={latitude}&lon={longitude}&start={start}&end={end}&appid={api_key}"
# Using open meteo 
base_url_history_weather = "https://archive-api.open-meteo.com/v1/archive?latitude=32.49&longitude=74.54&start_date=2026-02-27&end_date=2026-04-27&hourly=wind_speed_100m,precipitation,apparent_temperature,wind_gusts_10m,vapour_pressure_deficit,relative_humidity_2m&timezone=auto"
#features = ['main', 'wind', 'rain', 'visibility']

pollution_data_df = get_pollution_history(base_url_history_pollution=base_url_history_pollution)
weather_data_df = get_weather_history(base_url_history_weather=base_url_history_weather)

final_df = pd.merge(weather_data_df,pollution_data_df,"inner","datetime")
final_df["city"] = "Sialkot"
print(final_df.head(5))
print(len(final_df))

project = hopsworks.login()
fs = project.get_feature_store()

aqi_fg = fs.get_or_create_feature_group(
    name="sialkot_aqi_features",
    version=1,
    primary_key=["city"],         # The 'thing' we are measuring
    event_time="datetime",        # 'When' we measured it
    description="Hourly AQI and Open-Meteo weather features for Sialkot"
)

# 4. Insert the data!
aqi_fg.insert(final_df)