import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy import stats

class AnalyticsService:
    def __init__(self, config):
        self.config = config
    
    def analyze_consumption_patterns(self, usage_data: List[Dict]) -> Dict:
        """Analyze consumption patterns and provide insights"""
        if not usage_data:
            return self.get_sample_insights()
        
        # Convert to DataFrame
        df = pd.DataFrame(usage_data)
        
        # Calculate basic statistics
        total_usage = df['usage_kw'].sum()
        avg_daily = df.groupby('date')['usage_kw'].sum().mean()
        peak_usage = df['usage_kw'].max()
        avg_hourly = df.groupby('hour')['usage_kw'].mean()
        
        # Identify peak hours
        peak_hour = avg_hourly.idxmax()
        peak_value = avg_hourly.max()
        
        # Calculate variability
        daily_totals = df.groupby('date')['usage_kw'].sum()
        variability = daily_totals.std() / daily_totals.mean() * 100
        
        # Detect anomalies
        z_scores = np.abs(stats.zscore(daily_totals))
        anomalies = (z_scores > 2).sum()
        
        # Compare with similar households (benchmark)
        benchmark_avg = 30  # kWh/day benchmark
        efficiency_score = max(0, 100 - ((avg_daily - benchmark_avg) / benchmark_avg * 100))
        
        # Calculate carbon footprint
        carbon_footprint = total_usage * 0.92  # kg CO2 per kWh
        
        # Savings potential
        savings_potential = {
            "peak_shaving": peak_value * 0.2 * 30 * 0.15,  # Reduce peak by 20%
            "efficiency": total_usage * 0.1 * 0.15,  # 10% efficiency improvement
            "time_shift": total_usage * 0.15 * 0.15  # Shift 15% to off-peak
        }
        total_savings = sum(savings_potential.values())
        
        return {
            "summary": {
                "total_usage_kwh": round(total_usage, 2),
                "average_daily_kwh": round(avg_daily, 2),
                "peak_usage_kw": round(peak_usage, 2),
                "peak_hour": int(peak_hour),
                "efficiency_score": round(min(100, max(0, efficiency_score)), 1),
                "variability_percent": round(variability, 1),
                "anomaly_count": int(anomalies),
                "carbon_footprint_kg": round(carbon_footprint, 2)
            },
            "patterns": {
                "peak_hours": list(avg_hourly.nlargest(3).index),
                "offpeak_hours": list(avg_hourly.nsmallest(3).index),
                "weekend_vs_weekday": self.compare_weekend_weekday(df),
                "seasonal_trend": "increasing" if self.detect_trend(daily_totals) > 0 else "decreasing"
            },
            "recommendations": self.generate_recommendations(df),
            "savings": {
                "monthly_potential": round(total_savings, 2),
                "yearly_potential": round(total_savings * 12, 2),
                "breakdown": {k: round(v, 2) for k, v in savings_potential.items()}
            },
            "benchmarks": {
                "similar_homes": round(benchmark_avg, 2),
                "efficient_homes": round(benchmark_avg * 0.7, 2),
                "your_performance": round(avg_daily, 2)
            }
        }
    
    def compare_weekend_weekday(self, df: pd.DataFrame) -> Dict:
        """Compare weekend vs weekday usage"""
        df['date'] = pd.to_datetime(df['date'])
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        weekend_avg = df[df['is_weekend']]['usage_kw'].sum() / df['is_weekend'].sum() * 24
        weekday_avg = df[~df['is_weekend']]['usage_kw'].sum() / (~df['is_weekend']).sum() * 24
        
        return {
            "weekend_avg_kwh": round(weekend_avg, 2),
            "weekday_avg_kwh": round(weekday_avg, 2),
            "difference_percent": round((weekend_avg - weekday_avg) / weekday_avg * 100, 1)
        }
    
    def detect_trend(self, series: pd.Series) -> float:
        """Detect trend in time series data"""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series.values, 1)
        return slope
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Analyze hourly patterns
        hourly_avg = df.groupby('hour')['usage_kw'].mean()
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        
        if peak_hours:
            rec1 = {
                "id": "peak_shaving",
                "title": "Reduce Peak Hour Usage",
                "description": f"Your peak usage occurs at {peak_hours[0]}:00. Consider shifting high-energy activities.",
                "impact": "High",
                "savings": "$15-25/month",
                "effort": "Medium"
            }
            recommendations.append(rec1)
        
        # Check for nighttime usage
        night_usage = df[df['hour'].between(0, 5)]['usage_kw'].sum()
        if night_usage > df['usage_kw'].sum() * 0.1:  # More than 10% at night
            rec2 = {
                "id": "night_usage",
                "title": "Optimize Nighttime Consumption",
                "description": "Consider using smart plugs to turn off devices overnight.",
                "impact": "Medium",
                "savings": "$8-12/month",
                "effort": "Low"
            }
            recommendations.append(rec2)
        
        # General efficiency recommendations
        general_recs = [
            {
                "id": "smart_thermostat",
                "title": "Install Smart Thermostat",
                "description": "Optimize heating/cooling based on your schedule.",
                "impact": "High",
                "savings": "$20-35/month",
                "effort": "Medium"
            },
            {
                "id": "led_lighting",
                "title": "Switch to LED Lighting",
                "description": "Replace incandescent bulbs with energy-efficient LEDs.",
                "impact": "Medium",
                "savings": "$5-10/month",
                "effort": "Low"
            },
            {
                "id": "energy_audit",
                "title": "Schedule Free Energy Audit",
                "description": "Get personalized recommendations from our experts.",
                "impact": "High",
                "savings": "Varies",
                "effort": "Low"
            }
        ]
        
        recommendations.extend(general_recs[:2])
        
        return recommendations
    
    def get_sample_insights(self) -> Dict:
        """Return sample insights when no data is available"""
        return {
            "summary": {
                "total_usage_kwh": 450.5,
                "average_daily_kwh": 15.0,
                "peak_usage_kw": 3.2,
                "peak_hour": 19,
                "efficiency_score": 78.5,
                "variability_percent": 25.3,
                "anomaly_count": 2,
                "carbon_footprint_kg": 414.5
            },
            "patterns": {
                "peak_hours": [19, 20, 18],
                "offpeak_hours": [4, 5, 3],
                "weekend_vs_weekday": {
                    "weekend_avg_kwh": 16.2,
                    "weekday_avg_kwh": 14.5,
                    "difference_percent": 11.7
                },
                "seasonal_trend": "stable"
            },
            "recommendations": [
                {
                    "id": "peak_shaving",
                    "title": "Reduce Evening Usage",
                    "description": "Shift dishwasher and laundry to morning hours.",
                    "impact": "High",
                    "savings": "$18/month",
                    "effort": "Low"
                }
            ],
            "savings": {
                "monthly_potential": 28.50,
                "yearly_potential": 342.00,
                "breakdown": {
                    "peak_shaving": 18.00,
                    "efficiency": 6.75,
                    "time_shift": 3.75
                }
            },
            "benchmarks": {
                "similar_homes": 16.5,
                "efficient_homes": 11.5,
                "your_performance": 15.0
            }
        }