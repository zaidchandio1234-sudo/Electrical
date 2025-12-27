import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_dataset(output_path):
    """Create a sample electricity usage dataset"""
    print(f"Creating sample dataset at {output_path}")
    
    # Create date range
    start_date = datetime(2023, 1, 1)
    hours = 24 * 365  # One year of hourly data
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Generate realistic data
    np.random.seed(42)
    
    # Base patterns
    hours_sin = np.sin(np.arange(hours) * 2 * np.pi / 24)
    days_sin = np.sin(np.arange(hours) * 2 * np.pi / (24*7))
    seasons_sin = np.sin(np.arange(hours) * 2 * np.pi / (24*365))
    
    # Usage_kW: 0.5-4.0 kW with daily and weekly patterns
    usage = 2.0 + 0.5 * hours_sin + 0.3 * days_sin + 0.2 * seasons_sin + np.random.normal(0, 0.1, hours)
    usage = np.clip(usage, 0.5, 4.0)
    
    # Temperature: 15-35°C with seasonal pattern
    temperature = 25 + 10 * seasons_sin + 5 * np.sin(np.arange(hours) * 2 * np.pi / (24*30)) + np.random.normal(0, 2, hours)
    
    # Pressure: 1000-1020 hPa
    pressure = 1010 + 10 * np.sin(np.arange(hours) * 2 * np.pi / (24*5)) + np.random.normal(0, 1, hours)
    
    # Windspeed: 0-15 m/s
    windspeed = 5 + 5 * np.sin(np.arange(hours) * 2 * np.pi / (24*3)) + np.random.normal(0, 2, hours)
    windspeed = np.clip(windspeed, 0, 15)
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': timestamps,
        'Usage_kW': usage,
        'temperature': temperature,
        'pressure': pressure,
        'windspeed': windspeed
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Created sample dataset with {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df

if __name__ == "__main__":
    # Create dataset in the correct location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "electricity_usage.csv")
    create_sample_dataset(dataset_path)