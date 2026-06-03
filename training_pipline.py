import os
import hopsworks
import pandas as pd
import joblib
from dotenv import load_dotenv

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# 1. Load your API key
load_dotenv()
api_key = os.getenv("HopsworkAPI_KEY")

project = hopsworks.login(api_key_value=api_key)

fs = project.get_feature_store()

fg = fs.get_feature_group("sialkot_aqi_features", version=1)
df = fg.read()

# Dropping features not needed
# We drop columns that leak the answer, or that math can't understand
columns_to_drop = ['city', 'date', 'datetime', 'aqi', 'pm2_5', 'pm10']

X = df.drop(columns=columns_to_drop)

y = df['pm2_5']

# Use train_test_split to hide 20% of the data. Set random_state=42 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
# Initialize the competitors
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
    
# Ask the model to predict PM2.5 for the hidden X_test data
predictions = model.predict(X_test)
    
# Calculate the three required metrics by comparing 'predictions' to 'y_test'
rmse = root_mean_squared_error(y_test, predictions) 
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
    
results = {"RMSE": rmse, "MAE": mae, "R2": r2, "model_object": model}
print(f"Random Forest Regressor trained. R2 Score: {r2:.3f}")

# Push the best model to hopswork
# Save the model locally first as a .pkl file
model_dir = "aqi_model_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, f"{model_dir}/sialkot_pm25_model.pkl")

# Connect to the Model Registry
mr = project.get_model_registry()

# Create the cloud model entity, attaching our specific metrics
sialkot_model = mr.python.create_model(
    name="sialkot_pm2_5_predictor",
    metrics={
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    },
    description="Predicts PM2.5 using Random Forest Regressor"
)

# Upload the local directory to the cloud
sialkot_model.save(model_dir)