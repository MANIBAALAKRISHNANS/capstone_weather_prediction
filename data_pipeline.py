# data_pipeline.py
# Real-time data processing pipeline (mimics embedded systems)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("📊 DATA PROCESSING PIPELINE")
print("="*60 + "\n")

class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.original_count = 0
        self.cleaned_count = 0
    
    def clean_data(self, df):
        """Remove invalid readings (like embedded validation)"""
        print("🧹 Step 1: Cleaning Data...")
        
        self.original_count = len(df)
        
        # Remove NaN
        df = df.dropna()
        
        # Remove invalid temperature readings (valid range: -50 to 50°C)
        df = df[(df['temperature'] >= -50) & (df['temperature'] <= 50)]
        
        # Remove invalid humidity (0-100%)
        df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]
        
        # Remove invalid pressure (900-1050 hPa)
        df = df[(df['pressure'] >= 900) & (df['pressure'] <= 1050)]
        
        # Remove invalid wind speed (0-50 km/h)
        df = df[(df['wind_speed'] >= 0) & (df['wind_speed'] <= 50)]
        
        self.cleaned_count = len(df)
        removed = self.original_count - self.cleaned_count
        
        print(f"   ✓ Original records: {self.original_count}")
        print(f"   ✓ Removed outliers: {removed}")
        print(f"   ✓ Clean records: {self.cleaned_count}\n")
        
        return df
    
    def extract_features(self, df):
        """Extract features for ML models"""
        print("🔧 Step 2: Extracting Features...")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Rolling averages (6-hour, 12-hour moving average)
        df['temp_ma_6h'] = df['temperature'].rolling(window=6, min_periods=1).mean()
        df['temp_ma_12h'] = df['temperature'].rolling(window=12, min_periods=1).mean()
        df['humidity_ma_6h'] = df['humidity'].rolling(window=6, min_periods=1).mean()
        df['pressure_ma_6h'] = df['pressure'].rolling(window=6, min_periods=1).mean()
        
        # Lag features (1-hour, 6-hour, 24-hour)
        df['temp_lag_1h'] = df['temperature'].shift(1)
        df['temp_lag_6h'] = df['temperature'].shift(6)
        df['temp_lag_24h'] = df['temperature'].shift(24)
        df['humidity_lag_6h'] = df['humidity'].shift(6)
        
        # Rate of change
        df['temp_diff_1h'] = df['temperature'].diff(1)
        df['pressure_diff_1h'] = df['pressure'].diff(1)
        
        print(f"   ✓ Created {df.shape[1]} total features\n")
        
        return df
    
    def process(self, input_csv):
        """Complete processing pipeline"""
        
        # Load data
        print(f"📂 Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        print(f"   ✓ Loaded {len(df)} initial records\n")
        
        # Step 1: Clean
        df = self.clean_data(df)
        
        # Step 2: Extract features
        df = self.extract_features(df)
        
        # Step 3: Remove NaN created by shifting/rolling
        print("🔍 Step 3: Removing NaN values...")
        original_len = len(df)
        df = df.dropna()
        nan_count = original_len - len(df)
        print(f"   ✓ Removed NaN rows: {nan_count}")
        print(f"   ✓ Final records: {len(df)}\n")
        
        # Step 4: Create target variable (temperature 24 hours ahead)
        print("🎯 Step 4: Creating target variable...")
        df['temp_next_24h'] = df['temperature'].shift(-24)
        df = df.dropna()
        print(f"   ✓ Created target: temp_next_24h (Temperature +24 hours)")
        print(f"   ✓ Final records with target: {len(df)}\n")
        
        # Save processed data
        print("💾 Saving processed data...")
        df.to_csv('data/processed_data.csv', index=False)
        print(f"   ✓ Saved to: data/processed_data.csv\n")
        
        # Statistics
        print("📊 Final Data Statistics:")
        print(df[['temperature', 'humidity', 'pressure', 'wind_speed', 'temp_next_24h']].describe())
        
        print("\n" + "="*60 + "\n")
        
        return df


# Run the pipeline
if __name__ == "__main__":
    pipeline = DataPipeline()
    df = pipeline.process('data/weather_data.csv')
