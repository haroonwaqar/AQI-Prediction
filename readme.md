# AQI Forecaster

## Overview
The **Sialkot AQI Forecaster** is a Machine Learning Operations (MLOps) pipeline that predicts the Air Quality Index (AQI) for Sialkot, Pakistan, 72 hours into the future. 

Instead of a static notebook, this project operates as an autonomous, self-updating cloud system. It fetches live meteorological and chemical data, stores it in a serverless feature store, automatically retrains its predictive model to prevent data drift, and serves forecasts via a real-time web dashboard using the official US EPA AQI mathematical standards.

👉 **[View the Live Dashboard Here](https://aqi-prediction-sialkot.streamlit.app/)**

## System Architecture

The system is decoupled into three independent pipelines to ensure scalability and separation of concerns:

### 1. The Feature Pipeline (Data Ingestion)
* **Frequency:** Runs hourly via GitHub Actions.
* **Function:** Ingests live weather data (Open-Meteo) and chemical pollutants (OpenWeather), merges them asynchronously, and pushes the clean features to the **Hopsworks Feature Store**.

### 2. The Training Pipeline (Continuous Training)
* **Frequency:** Runs weekly (Sundays at 00:45 UTC) via GitHub Actions.
* **Function:** Downloads the freshest historical data from Hopsworks. It forces an algorithmic bake-off (Ridge, Random Forest, XGBoost), dynamically selects the best performer, and pushes the serialized `.pkl` brain to the **Hopsworks Model Registry**.
* **Model:** Random Forest Regressor ($R^2 \approx 0.819$).

### 3. The Inference Pipeline (Web UI)
* **Environment:** Streamlit Community Cloud.
* **Function:** Downloads the model from the cloud, fetches a 72-hour future weather forecast, predicts continuous PM2.5 values, and translates them into actionable US EPA AQI health alerts for end users.

## Tech Stack
* **Machine Learning:** `scikit-learn`, `xgboost`, `pandas`
* **MLOps & Storage:** `Hopsworks`
* **CI/CD Automation:** `GitHub Actions`
* **Front-End / UI:** `streamlit`
* **APIs:** `Open-Meteo`, `OpenWeather`

## Features & UX
* US AQI Standard: Uses exact linear interpolation to convert raw PM2.5 predictions into the standardized 0-500 scale.
* Proactive Alerts: Evaluates the 3-day forecast and triggers dynamic UI banners (Warning/Error) if hazardous smog conditions are predicted.
* Caching: Implements cache to ensure the model and API payloads remain in memory, providing sub-second load times for users.
