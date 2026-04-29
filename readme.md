# AQI Prediction
A forecast Air Quality Index (AQI) levels over a 3-day period using a 100% serverless stack.

## Feature Pipeline:
It fetches raw weather and pollutant data from external APIs OpenWeather and Open Meteo.
The script computes features, such as time-based metrics and AQI change rates, and stores them in a feature Store in Hopswork.

Historical data is backfilled to generate training sets for machine learning models.