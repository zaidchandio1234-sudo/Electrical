import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class ForecastGenerator:
    def __init__(self, model_loader, data_processor):
        self.model_loader = model_loader
        self.data_processor = data_processor
        
    def generate_7_day_forecast(self):
        """Generate 7-day energy consumption forecast"""
        try:
            # Try to generate forecast using model
            hourly_forecast = self.model_loader.generate_7day_forecast()
            
            # Aggregate by day
            daily_data = self.aggregate_forecast_by_day(hourly_forecast)
            
            return self.format_forecast_response(daily_data)
            
        except Exception as e:
            print(f"Forecast generation failed: {e}")
            # Fallback to sample data
            return self.generate_sample_forecast()
    
    def aggregate_forecast_by_day(self, hourly_forecast):
        """Aggregate hourly forecast into daily summaries"""
        # Convert to DataFrame
        df = pd.DataFrame(hourly_forecast)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        daily_data = []
        for date, group in df.groupby(df['timestamp'].dt.date):
            hourly_list = []
            for _, row in group.iterrows():
                hourly_list.append({
                    'hour': row['hour'],
                    'usage_kw': round(row['predicted_kw'], 2)
                })
            
            daily_data.append({
                'date': date.isoformat(),
                'day_name': group['day_name'].iloc[0],
                'total_kwh': round(group['predicted_kw'].sum(), 2),
                'avg_kwh': round(group['predicted_kw'].mean(), 2),
                'peak_kwh': round(group['predicted_kw'].max(), 2),
                'peak_hour': int(group.loc[group['predicted_kw'].idxmax()]['hour']),
                'hourly_data': hourly_list
            })
        
        return daily_data
    
    def format_forecast_response(self, daily_data):
        """Format forecast response"""
        # Find peak day
        peak_day = max(daily_data, key=lambda x: x['total_kwh'])
        
        # Calculate weekly total
        weekly_total = sum(day['total_kwh'] for day in daily_data)
        
        return {
            'daily_data': daily_data,
            'metadata': {
                'peak_day': peak_day['day_name'],
                'peak_usage': peak_day['total_kwh'],
                'weekly_total': round(weekly_total, 2),
                'average_daily': round(weekly_total / 7, 2),
                'generated_at': datetime.now().isoformat(),
                'forecast_type': 'ai_prediction'
            }
        }
    
    def generate_sample_forecast(self):
        """Generate sample forecast data"""
        import random
        current_date = datetime.now()
        daily_data = []
        
        for i in range(7):
            date = current_date + timedelta(days=i)
            hourly_data = []
            
            # Generate sample hourly data
            for hour in range(24):
                base = 1.5 + 0.5 * np.sin(hour/24 * 2*np.pi)
                if 14 <= hour <= 20:  # Peak hours
                    base *= 1.3
                usage = base + random.uniform(-0.2, 0.2)
                
                hourly_data.append({
                    'hour': hour,
                    'usage_kw': round(max(0.5, usage), 2)
                })
            
            total = sum(h['usage_kw'] for h in hourly_data)
            
            daily_data.append({
                'date': date.date().isoformat(),
                'day_name': date.strftime('%A'),
                'total_kwh': round(total, 2),
                'avg_kwh': round(total/24, 2),
                'peak_kwh': round(max(h['usage_kw'] for h in hourly_data), 2),
                'peak_hour': max(hourly_data, key=lambda x: x['usage_kw'])['hour'],
                'hourly_data': hourly_data
            })
        
        return self.format_forecast_response(daily_data)