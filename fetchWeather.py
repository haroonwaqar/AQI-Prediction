import requests
import pandas as pd

# Function to get historical pollution data via openWeather
def get_weather_history(base_url_history_weather):
    response_weather = requests.get(url=base_url_history_weather)
    weather = response_weather.json()
    weather = weather["hourly"]
    weather_data = pd.DataFrame(weather)
    weather_data.rename(columns={'time':'datetime'}, inplace=True)
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    
    return weather_data
