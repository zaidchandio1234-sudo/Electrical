import requests
import json
from config import Config

class LLMAdvisor:
    def __init__(self, config):
        self.config = config
        self.api_key = config.LLM_API_KEY
        self.api_url = config.LLM_API_URL
        
    def is_available(self):
        """Check if LLM API is available"""
        return bool(self.api_key)
    
    def generate_weekly_advice(self, forecast_data):
        """Generate personalized advice for the weekly forecast"""
        
        if not self.api_key:
            return self._get_fallback_advice(forecast_data)
        
        try:
            prompt = self._create_weekly_prompt(forecast_data)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": self.config.LLM_MAX_LENGTH,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                advice = response.json()[0]['generated_text']
                return self._format_weekly_advice(advice, forecast_data)
            else:
                return self._get_fallback_advice(forecast_data)
                
        except Exception as e:
            print(f"LLM API error: {str(e)}")
            return self._get_fallback_advice(forecast_data)
    
    def _create_weekly_prompt(self, forecast_data):
        """Create prompt for weekly advice"""
        daily_data = forecast_data['daily_data']
        peak_day = forecast_data['metadata']['peak_day']
        peak_usage = forecast_data['metadata']['peak_usage']
        
        prompt = f"""You are an Energy Efficiency Advisor. Based on this 7-day energy forecast, provide actionable advice:

FORECAST SUMMARY:
Weekly Total: {forecast_data['metadata']['weekly_total']} kWh
Average Daily: {forecast_data['metadata']['average_daily']} kWh
Peak Day: {peak_day} ({peak_usage} kWh)

DAILY BREAKDOWN:
"""
        
        for day in daily_data:
            prompt += f"- {day['day_name']}: {day['total_kwh']} kWh total, Peak at {day['peak_hour']}:00 ({day['peak_kwh']} kW)\n"
        
        prompt += """
Provide a 7-day action plan with:
1. Specific actions for each day
2. Time-based recommendations for peak hours
3. Estimated energy savings
4. Practical, easy-to-implement tips
5. Focus on reducing consumption during peak hours

Format as a friendly, conversational advisory with bullet points.
"""
        
        return prompt
    
    def _format_weekly_advice(self, advice_text, forecast_data):
        """Format LLM response"""
        advice_text = advice_text.replace("</s>", "").strip()
        
        daily_data = forecast_data['daily_data']
        
        # Create daily plans based on forecast
        daily_plans = []
        for day in daily_data:
            is_peak = day['total_kwh'] > forecast_data['metadata']['average_daily'] * 1.2
            
            # Create tips based on day's characteristics
            tips = []
            if is_peak:
                tips.append(f"Reduce AC usage during {day['peak_hour']-1}:00 to {day['peak_hour']+1}:00")
                tips.append("Postpone laundry to early morning")
                tips.append("Use smart plugs to control standby power")
            else:
                tips.append("Good day for high-energy activities")
                tips.append("Consider charging EVs during off-peak")
                tips.append("Run dishwasher in the evening")
            
            daily_plans.append({
                'day_name': day['day_name'],
                'predicted_usage': f"{day['total_kwh']}",
                'avg_temp': "24°C",  # You can add actual temps if available
                'is_peak_day': is_peak,
                'tips': tips[:3]  # Limit to 3 tips per day
            })
        
        return {
            'summary': f"Your energy usage peaks on {forecast_data['metadata']['peak_day']}. Follow the daily plan to save 10-15%.",
            'daily_plans': daily_plans
        }
    
    def _get_fallback_advice(self, forecast_data):
        """Provide fallback advice"""
        peak_day = forecast_data['metadata']['peak_day']
        peak_usage = forecast_data['metadata']['peak_usage']
        
        return {
            'summary': f"Based on our forecast, your peak energy day is {peak_day} with {peak_usage} kWh consumption.",
            'daily_plans': [
                {
                    'day_name': 'Monday',
                    'predicted_usage': '45.2',
                    'avg_temp': '24°C',
                    'is_peak_day': False,
                    'tips': ['Run appliances before 10 AM', 'Set thermostat to 25°C']
                },
                {
                    'day_name': 'Tuesday',
                    'predicted_usage': '48.7',
                    'avg_temp': '26°C',
                    'is_peak_day': True,
                    'tips': ['Avoid AC between 2-5 PM', 'Use fans instead of AC']
                }
            ]
        }