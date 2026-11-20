import pandas as pd
import requests

# 1. Define location and 1-year date range
# Example coordinates: Bengaluru (12.9716, 77.5946), Delhi (28.6139, 77.2090), London (51.5074, -0.1278)
LATITUDE = 12.9716
LONGITUDE = 77.5946
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

print("Fetching historical weather data...")
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "precipitation",
    ],
    "timezone": "auto",
}
res_weather = requests.get(weather_url, params=weather_params).json()
df_weather = pd.DataFrame(res_weather["hourly"])
df_weather["time"] = pd.to_datetime(df_weather["time"])

print("Fetching historical air pollution data...")
aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ["pm2_5", "pm10", "us_aqi"],
    "timezone": "auto",
}
res_aq = requests.get(aq_url, params=aq_params).json()
df_aq = pd.DataFrame(res_aq["hourly"])
df_aq["time"] = pd.to_datetime(df_aq["time"])

# 2. Merge on timestamp (Inner Join ensures exact time alignment)
df_final = pd.merge(df_weather, df_aq, on="time", how="inner")

# 3. Clean and verify missing values
df_final = df_final.dropna().reset_index(drop=True)

# 4. Save to CSV for ML pipeline
csv_filename = "city_weather_pollution_2024.csv"
df_final.to_csv(csv_filename, index=False)

print(f"\nDataset successfully generated: {csv_filename}")
print(f"Total instances collected: {len(df_final)} rows")
print("\nFirst 3 rows:")
print(df_final.head(3))