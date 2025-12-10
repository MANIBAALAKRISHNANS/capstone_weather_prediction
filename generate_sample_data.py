# generate_sample_data.py
# Generates 1 year of synthetic weather data for training

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("\n" + "="*60)
print("📊 WEATHER DATA GENERATOR")
print("="*60 + "\n")

# Generate 1 year of hourly weather data
dates = pd.date_range(start='2022-01-01', periods=8760, freq='H')

print(f"Generating {len(dates)} hourly records from 2022-01-01...")

# Create realistic weather patterns
temperature = 20 + 10*np.sin(np.arange(8760)*2*np.pi/8760) + np.random.normal(0, 2, 8760)
humidity = np.clip(60 + 20*np.sin(np.arange(8760)*2*np.pi/8760) + np.random.normal(0, 5, 8760), 0, 100)
pressure = 1013 + 5*np.sin(np.arange(8760)*2*np.pi/8760) + np.random.normal(0, 1, 8760)
wind_speed = np.clip(10 + 5*np.sin(np.arange(8760)*2*np.pi/8760) + np.random.normal(0, 1, 8760), 0, 50)
rainfall = np.random.exponential(scale=2, size=8760)

data = {
    'timestamp': dates,
    'temperature': temperature,
    'humidity': humidity,
    'pressure': pressure,
    'wind_speed': wind_speed,
    'rainfall': rainfall
}

df = pd.DataFrame(data)

# Save
df.to_csv('data/weather_data.csv', index=False)

print(f"✓ Generated {len(df)} weather records")
print(f"✓ Saved to: data/weather_data.csv\n")
print("Data Sample:")
print(df.head(10))
print(f"\nData Statistics:")
print(df.describe())
print("\n" + "="*60 + "\n")
