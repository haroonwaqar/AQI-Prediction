# Sialkot AQI Forecaster

An autonomous, production-grade MLOps system that predicts Air Quality Index (AQI) for Sialkot, Pakistan, 72 hours into the future powered by a 100% serverless stack.

**[View the Live Dashboard](https://aqi-prediction-sialkot.streamlit.app/)**

## Project Overview

The **Sialkot AQI Forecaster** is a full end-to-end Machine Learning Operations (MLOps) pipeline. Instead of a static notebook, this system operates as a self-updating cloud service. It:

1. **Ingests** live hourly meteorological and chemical pollutant data from external APIs
2. **Engineers** features and stores them in a serverless Feature Store (Hopsworks)
3. **Trains** a champion ML model daily and registers it to a Model Registry
4. **Serves** 72-hour PM2.5 predictions via a Streamlit dashboard, with automatic conversion to the official **US EPA AQI scale** and dynamic health alerts

The target variable is continuous **PM2.5 concentration (µg/m³)**, which is then converted to the 0–500 AQI scale using the official EPA linear interpolation formula giving users a precise, standardized air quality reading rather than a coarse category.

## System Architecture

The system is decoupled into three independent pipelines to ensure scalability and separation of concerns:

### 1. The Feature Pipeline (Data Ingestion)
* **Frequency:** Runs hourly via GitHub Actions.
* **Function:** Ingests live weather data (Open-Meteo) and chemical pollutants (OpenWeather), merges them asynchronously, and pushes the clean features to the **Hopsworks Feature Store**.

### 2. The Training Pipeline (Continuous Training)
* **Frequency:** Runs daily (At 00:45 UTC) via GitHub Actions.
* **Function:** Downloads the freshest historical data from Hopsworks. It forces an algorithmic bake-off (Ridge, Random Forest, XGBoost), dynamically selects the best performer, and pushes the serialized `.pkl` brain to the **Hopsworks Model Registry**.
* **Model:** Best model was Xgboost Regressor

### 3. The Inference Pipeline (Web UI)
* **Environment:** Streamlit Community Cloud.
* **Function:** Downloads the model from the cloud, fetches a 72-hour future weather forecast, predicts continuous PM2.5 values, and translates them into actionable US EPA AQI health alerts for end users.

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | `scikit-learn`, `xgboost`, `pandas`, `numpy`, `SHAP` |
| MLOps & Feature Store | `Hopsworks` (serverless) |
| Model Registry | `Hopsworks Model Registry` |
| CI/CD Automation | `GitHub Actions` |
| Web App / Dashboard | `Streamlit Community Cloud` |
| Weather API | `Open-Meteo` (free, no key required) |
| Pollutant API | `OpenWeatherMap Air Pollution API` |
| Serialization | `joblib` (.pkl) |
| Timezone handling | `pytz` |

## Feature Pipeline

**File:** `feature_pipeline.py`
**Runs:** Every hour via GitHub Actions

### What it does

1. **Fetches meteorological data** from the Open-Meteo archive/forecast API for Sialkot (lat: 32.49, lon: 74.54), pulling: wind_speed_100m, precipitation, apparent_temperature, wind_gusts_10m, vapour_pressure_deficit, relative_humidity_2m
2. **Fetches chemical pollutant data** from the OpenWeatherMap Air Pollution API, pulling: co, no2, o3, so2, pm2_5, pm10, nh3, aqi
3. **Merges** both sources on the `datetime` column using an inner join
4. **Pushes** the clean merged dataframe to the Hopsworks Feature Group `sialkot_aqi_features_v2`

## Training Pipeline

**File:** `training_pipeline.py`
**Runs:** Daily via GitHub Actions

### What it does

1. **Fetches** the full historical feature set from the Hopsworks Feature Group
2. **Enforces a chronological 80/20 split** using `shuffle=False` to prevent temporal data leakage
3. **Trains XGBoost** with tuned hyperparameters
4. **Evaluates** on the held-out future test set using RMSE, MAE, and R²
5. **Serializes** the model to a `.pkl` file and registers it to the Hopsworks Model Registry with metrics attached

## Inference Pipeline (Web App)

**File:** `app.py`
**Environment:** Streamlit Community Cloud

### What it does

1. Downloads the latest production model from the Hopsworks Model Registry (`@st.cache_resource`)
2. Fetches a 5-day hourly weather forecast from Open-Meteo and a 5-day pollutant forecast from OpenWeatherMap (`@st.cache_data(ttl=3600)`)
3. Applies identical feature engineering to the inference data as was applied during training
4. Generates 72 hourly PM2.5 predictions
5. Converts each prediction to the US EPA AQI scale
6. Displays a 72-hour PM2.5 line chart and a 3-day daily AQI summary dashboard
7. Fires dynamic health alerts based on the worst-case predicted AQI

### Dashboard components

- **72-hour PM2.5 line chart** — continuous forecast visualization
- **3-day daily AQI cards** — date, AQI level label, emoji, daily average PM2.5
- **Health alert banners** — automatically triggered at AQI thresholds:
  - All Clear (AQI < 101)
  - Advisory (AQI 101–150, sensitive groups)
  - Alert (AQI ≥ 151, hazardous conditions)
- **Raw hourly data table** — full `datetime`, `Predicted_PM25`, `Predicted_AQI` output

## CI/CD Automation

Two GitHub Actions workflows automate the entire system:

### Feature pipeline workflow
```yaml
# Runs every hour
schedule:
  - cron: '0 * * * *'
```
Installs dependencies from `requirements.txt`, injects secrets (`WeatherAPI_KEY`, `HopsworkAPI_KEY`), and executes `feature_pipeline.py`.

### Training pipeline workflow
```yaml
# Runs daily at 00:45 UTC
schedule:
  - cron: '45 0 * * *'
```
Installs dependencies from `requirements.txt` and executes `training_pipeline.py`. The daily schedule ensures the model continuously incorporates fresh data, combating concept drift as Sialkot's air quality evolves seasonally.

## Key design decisions

**No PM2.5 lags as features.** Unlike a 1-step-ahead forecast, this system predicts 72 hours into the future. Using `pm2_5_lag1` would be valid for 1-hour-ahead prediction but would require a recursive forecasting loop for 72-hour horizons (propagating prediction errors across all steps). The decision was made to use a direct multi-step approach with weather-based features only, which are available from Open-Meteo's free forecast API.

**Cyclical hour encoding.** Converting the 24-hour clock to sine/cosine pairs (rather than a raw integer 0–23) allows the model to correctly understand that hour 23 and hour 0 are adjacent. This is critical for capturing daily traffic cycles and nocturnal temperature inversions that drive Sialkot's PM2.5 peaks.

**Lags on meteorological variables only.** `temp_lag1` and `wind_rolling_6` are derived from weather, which is available at inference time via the forecast API — unlike PM2.5, which is unknown in the future.

## Project Structure

```
AQI-Prediction/
│
├── .github/
│   └── workflows/
│       ├── hourly_pipeline.yml       # Runs hourly — fetches & stores features
│       └── retrain_model.yml         # Runs daily — trains & registers model
│
├── notebooks/                        # EDA, model experiments
│   └── SampleFetch.ipynb    
|   └── eda_features.ipynb
|   └── model_training.ipynb
│
├── scripts/                          # Utility and testing scripts
│   └── fetchPollution.py
|   └── fetchWeather.py
|   └── test_pipeline.py
│
├── app.py                            # Streamlit inference dashboard
├── feature_pipeline.py               # Hourly data ingestion pipeline
├── training_pipeline.py              # Daily model training pipeline
│
├── requirements.txt                  # Streamlit app dependencies
│
├── .gitignore
└── readme.md
```

## Local Setup

### Prerequisites
- Python 3.2
- A free [Hopsworks account](https://app.hopsworks.ai/)
- A free [OpenWeatherMap API key](https://openweathermap.org/api)

### Installation

```bash
git clone https://github.com/haroonwaqar/AQI-Prediction.git
cd AQI-Prediction
pip install -r requirements.txt
```

### Running the feature pipeline locally

```bash
# Create a .env file with your API keys (see Environment Variables below)
python feature_pipeline.py
```

### Running the training pipeline locally

```bash
python training_pipeline.py
```

### Running the Streamlit app locally

```bash
streamlit run app.py
```

## About

The system demonstrates a production-grade MLOps architecture using entirely free, serverless infrastructure using no paid cloud resources required.

**Live dashboard:** https://aqi-prediction-sialkot.streamlit.app/

**Repository:** https://github.com/haroonwaqar/AQI-Prediction