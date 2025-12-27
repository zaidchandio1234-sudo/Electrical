import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config

class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def prepare_forecast_data(self):
        """Load and prepare data for forecasting"""
        try:
            # Load your electricity usage data
            df = pd.read_csv(self.config.DATA_PATH)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            
            # Extract features
            features = df[self.config.FEATURES].values
            
            return features
            
        except Exception as e:
            raise Exception(f"Failed to prepare data: {str(e)}")
    
    def aggregate_forecast_by_day(self, forecast_data):
        """Aggregate hourly forecast into daily summaries"""
        # Convert to DataFrame
        df = pd.DataFrame(forecast_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by day
        daily_data = []
        for date, group in df.groupby(df['timestamp'].dt.date):
            daily_data.append({
                'date': date.isoformat(),
                'day_name': group['day_name'].iloc[0],
                'total_kwh': round(group['predicted_kw'].sum(), 2),
                'avg_kwh': round(group['predicted_kw'].mean(), 2),
                'peak_kwh': round(group['predicted_kw'].max(), 2),
                'peak_hour': int(group.loc[group['predicted_kw'].idxmax()]['hour']),
                'hourly_data': [
                    {
                        'hour': row['hour'],
                        'usage_kw': round(row['predicted_kw'], 2)
                    }
                    for _, row in group.iterrows()
                ]
            })
        
        return daily_data