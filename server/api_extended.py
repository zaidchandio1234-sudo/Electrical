from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

from auth import AuthService, get_current_user
from billing import BillingService, BillingPeriod
from analytics import AnalyticsService
from config import Config
from model_loader import ModelLoader
from llm_advisor import LLMAdvisor

# Initialize services
config = Config()
auth_service = AuthService(config)
billing_service = BillingService(config)
analytics_service = AnalyticsService(config)
model_loader = ModelLoader(config)
llm_advisor = LLMAdvisor(config)


# Add these imports at the top
from auth import AuthService, UserRegister, UserLogin, Token, auth_service
from fastapi.security import HTTPBearer

# Add authentication endpoints
@router.post("/auth/register", response_model=Token) # type: ignore
async def register_user(user_data: UserRegister):
    """Register new user"""
    try:
        result = await auth_service.register_user(user_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/auth/login", response_model=Token) # type: ignore
async def login_user(login_data: UserLogin):
    """User login"""
    try:
        result = await auth_service.login_user(login_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.get("/auth/profile") # type: ignore
async def get_user_profile(current_user: Dict = Depends(auth_service.get_current_user)):
    """Get user profile"""
    return {
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "name": current_user["name"],
        "account_type": current_user["account_type"]
    }

# Protect other endpoints with authentication
@router.get("/dashboard/summary") # type: ignore
async def get_dashboard_summary(current_user: Dict = Depends(auth_service.get_current_user)):
    """Get dashboard summary (protected)"""
    # ... existing code ...

@router.post("/forecast/generate") # type: ignore
async def generate_forecast(current_user: Dict = Depends(auth_service.get_current_user)):
    """Generate forecast (protected)"""
    # ... existing code ...
# Create router
router = APIRouter(prefix="/api/v1", tags=["enterprise"])

# Request/Response Models
class LoginRequest(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    email: str
    name: str
    address: Optional[str]
    meter_id: Optional[str]
    account_type: str
    joined_date: str

class Device(BaseModel):
    id: str
    name: str
    type: str
    status: str
    power_usage: float
    last_active: str

class Alert(BaseModel):
    id: str
    type: str
    severity: str
    message: str
    timestamp: str
    read: bool

# Authentication endpoints
@router.post("/auth/login")
async def login(request: LoginRequest):
    """User login endpoint"""
    user = await auth_service.authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = auth_service.create_access_token(data={"sub": user["email"]})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user["name"]
    }

@router.get("/auth/profile")
async def get_profile(current_user: Dict = Depends(auth_service.get_current_user)):
    """Get user profile"""
    return {
        "email": current_user["email"],
        "name": current_user["name"],
        "account_type": current_user["account_type"],
        "meter_id": "MTR-001-2024",
        "address": "123 Green Street, Eco City",
        "joined_date": "2024-01-15",
        "account_status": "active",
        "payment_method": "credit_card_ending_4242"
    }

# Dashboard endpoints
@router.get("/dashboard/summary")
async def get_dashboard_summary(current_user: Dict = Depends(auth_service.get_current_user)):
    """Get dashboard summary data"""
    # Generate sample data for dashboard
    import random
    
    current_time = datetime.now()
    hour = current_time.hour
    
    # Real-time usage simulation
    base_usage = 1.5 + 0.5 * random.uniform(-0.2, 0.2)
    if 14 <= hour <= 20:  # Peak hours
        base_usage *= 1.3
    
    # Today's usage simulation
    today_usage = 0
    for h in range(24):
        hour_usage = 1.5 + 0.5 * random.uniform(-0.2, 0.2)
        if 14 <= h <= 20:
            hour_usage *= 1.3
        today_usage += hour_usage
    
    # This month's usage
    month_usage = today_usage * 15 + random.uniform(-10, 10)  # Approximate
    
    return {
        "real_time": {
            "current_usage_kw": round(base_usage, 2),
            "voltage": 240,
            "frequency": 60,
            "power_factor": 0.95
        },
        "today": {
            "total_kwh": round(today_usage, 2),
            "cost": round(today_usage * 0.15, 2),
            "peak_kw": round(base_usage * 1.2, 2),
            "average_kw": round(today_usage / 24, 2)
        },
        "month": {
            "total_kwh": round(month_usage, 2),
            "projected_cost": round(month_usage * 0.15, 2),
            "days_remaining": 15,
            "daily_average": round(month_usage / 15, 2)
        },
        "efficiency": {
            "score": 78,
            "compared_to_neighbors": "better",
            "percentile": 65,
            "trend": "improving"
        }
    }

@router.get("/dashboard/usage/hourly")
async def get_hourly_usage(days: int = 1, current_user: Dict = Depends(auth_service.get_current_user)):
    """Get hourly usage data"""
    import numpy as np
    
    data = []
    current_time = datetime.now()
    
    for hour_offset in range(24 * days):
        hour_time = current_time - timedelta(hours=hour_offset) # type: ignore
        hour = hour_time.hour
        
        # Generate realistic usage pattern
        base = 1.5 + 0.5 * np.sin(hour/24 * 2*np.pi)
        if 14 <= hour <= 20:  # Peak hours
            base *= 1.3
        
        usage = base + np.random.normal(0, 0.1)
        
        data.append({
            "timestamp": hour_time.isoformat(),
            "hour": hour,
            "usage_kw": round(max(0.5, usage), 2),
            "is_peak": 14 <= hour <= 20,
            "temperature": 25 + 5 * np.sin(hour/24 * 2*np.pi) + np.random.normal(0, 2)
        })
    
    return {
        "user_id": current_user["user_id"],
        "data": data[::-1],  # Reverse to chronological order
        "period": f"last_{days}_days",
        "total_kwh": round(sum(d['usage_kw'] for d in data) / 1000, 2)
    }

# Billing endpoints
@router.get("/billing/current")
async def get_current_bill(current_user: Dict = Depends(auth_service.get_current_user)):
    """Get current bill information"""
    current_bill = billing_service.get_current_bill(current_user["user_id"])
    
    if not current_bill:
        # Generate new bill
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_bill = billing_service.generate_monthly_bill(
            current_user["user_id"], current_month, current_year
        )
    
    return current_bill

@router.get("/billing/history")
async def get_billing_history(limit: int = 12, current_user: Dict = Depends(auth_service.get_current_user)):
    """Get billing history"""
    history = billing_service.get_billing_history(current_user["user_id"], limit)
    return {
        "user_id": current_user["user_id"],
        "history": history,
        "total_count": len(history)
    }

@router.post("/billing/pay")
async def process_payment(bill_id: int, payment_method: str = "credit_card", 
                         current_user: Dict = Depends(auth_service.get_current_user)):
    """Process payment for a bill"""
    success = billing_service.process_payment(current_user["user_id"], bill_id, payment_method)
    
    if not success:
        raise HTTPException(status_code=400, detail="Payment failed")
    
    return {
        "success": True,
        "message": "Payment processed successfully",
        "bill_id": bill_id,
        "paid_at": datetime.now().isoformat()
    }

# Analytics endpoints
@router.get("/analytics/insights")
async def get_analytics_insights(days: int = 30, current_user: Dict = Depends(auth_service.get_current_user)):
    """Get consumption analytics and insights"""
    # Generate sample usage data
    import numpy as np
    
    usage_data = []
    current_date = datetime.now()
    
    for day_offset in range(days):
        date = current_date - timedelta(days=day_offset) # type: ignore
        date_str = date.date().isoformat()
        
        for hour in range(24):
            # Daily pattern
            base = 1.5 + 0.5 * np.sin(hour/24 * 2*np.pi)
            
            # Weekend effect
            if date.weekday() >= 5:  # Weekend
                base *= 1.2
            
            # Random variation
            usage = base + np.random.normal(0, 0.2)
            
            usage_data.append({
                "hour": hour,
                "usage_kw": max(0.5, usage),
                "date": date_str,
                "is_peak": 14 <= hour <= 20
            })
    
    insights = analytics_service.analyze_consumption_patterns(usage_data)
    
    return {
        "user_id": current_user["user_id"],
        "period_days": days,
        "insights": insights
    }

@router.get("/analytics/recommendations")
async def get_recommendations(current_user: Dict = Depends(auth_service.get_current_user)):
    """Get personalized energy recommendations"""
    # Get insights first
    insights_response = await get_analytics_insights(30, current_user)
    insights = insights_response["insights"]
    
    # Enhance with AI recommendations
    ai_recommendations = llm_advisor.generate_weekly_advice({
        "daily_data": [],
        "metadata": {
            "peak_day": "Monday",
            "peak_usage": insights["summary"]["peak_usage_kw"],
            "weekly_total": insights["summary"]["total_usage_kwh"],
            "average_daily": insights["summary"]["average_daily_kwh"]
        }
    })
    
    return {
        "user_id": current_user["user_id"],
        "generated_at": datetime.now().isoformat(),
        "insight_based": insights["recommendations"],
        "ai_recommendations": ai_recommendations,
        "priority_order": ["peak_shaving", "smart_thermostat", "energy_audit"]
    }

# Smart Devices endpoints
@router.get("/devices")
async def get_devices(current_user: Dict = Depends(auth_service.get_current_user)):
    """Get list of smart devices"""
    # Sample devices data
    devices = [
        {
            "id": "dev-001",
            "name": "Living Room Thermostat",
            "type": "thermostat",
            "status": "active",
            "power_usage": 0.45,
            "last_active": datetime.now().isoformat(),
            "room": "Living Room",
            "manufacturer": "Nest",
            "model": "Learning Thermostat"
        },
        {
            "id": "dev-002",
            "name": "Kitchen Lights",
            "type": "lighting",
            "status": "active",
            "power_usage": 0.12,
            "last_active": datetime.now().isoformat(),
            "room": "Kitchen",
            "manufacturer": "Philips Hue",
            "model": "Color Ambiance"
        },
        {
            "id": "dev-003",
            "name": "Smart Plug - TV",
            "type": "outlet",
            "status": "idle",
            "power_usage": 0.08,
            "last_active": (datetime.now() - timedelta(hours=2)).isoformat(), # type: ignore
            "room": "Living Room",
            "manufacturer": "TP-Link",
            "model": "Kasa Smart Plug"
        },
        {
            "id": "dev-004",
            "name": "Refrigerator",
            "type": "appliance",
            "status": "active",
            "power_usage": 1.25,
            "last_active": datetime.now().isoformat(),
            "room": "Kitchen",
            "manufacturer": "LG",
            "model": "Smart InstaView"
        }
    ]
    
    return {
        "user_id": current_user["user_id"],
        "total_devices": len(devices),
        "active_devices": sum(1 for d in devices if d["status"] == "active"),
        "total_power": round(sum(d["power_usage"] for d in devices), 2),
        "devices": devices
    }

# Alerts endpoints
@router.get("/alerts")
async def get_alerts(unread_only: bool = False, current_user: Dict = Depends(auth_service.get_current_user)):
    """Get user alerts"""
    # Sample alerts
    alerts = [
        {
            "id": "alert-001",
            "type": "usage_spike",
            "severity": "warning",
            "message": "Unusual energy consumption detected in kitchen circuit",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(), # type: ignore
            "read": False,
            "action_required": True,
            "suggested_action": "Check appliance status"
        },
        {
            "id": "alert-002",
            "type": "peak_hour",
            "severity": "info",
            "message": "Peak pricing period starts in 30 minutes",
            "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(), # type: ignore
            "read": True,
            "action_required": False,
            "suggested_action": "Delay high-energy tasks"
        },
        {
            "id": "alert-003",
            "type": "savings_achieved",
            "severity": "success",
            "message": "You've saved 15% compared to last week!",
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(), # type: ignore
            "read": True,
            "action_required": False,
            "suggested_action": "View savings report"
        },
        {
            "id": "alert-004",
            "type": "device_offline",
            "severity": "warning",
            "message": "Smart plug in bedroom has been offline for 4 hours",
            "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(), # type: ignore
            "read": False,
            "action_required": True,
            "suggested_action": "Check device connection"
        }
    ]
    
    if unread_only:
        alerts = [alert for alert in alerts if not alert["read"]]
    
    return {
        "user_id": current_user["user_id"],
        "total_alerts": len(alerts),
        "unread_count": sum(1 for a in alerts if not a["read"]),
        "alerts": alerts
    }

@router.post("/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: str, current_user: Dict = Depends(auth_service.get_current_user)):
    """Mark alert as read"""
    # In a real implementation, this would update database
    return {
        "success": True,
        "message": f"Alert {alert_id} marked as read",
        "alert_id": alert_id
    }

# Forecast endpoints
@router.post("/forecast/generate")
async def generate_forecast(current_user: Dict = Depends(auth_service.get_current_user)):
    """Generate 7-day energy forecast"""
    try:
        # Use existing forecast generator
        from forecast_generator import ForecastGenerator
        from data_processor import DataProcessor
        
        data_processor = DataProcessor(config)
        forecast_generator = ForecastGenerator(model_loader, data_processor)
        
        forecast_data = forecast_generator.generate_7_day_forecast()
        
        # Add user context
        forecast_data["user_id"] = current_user["user_id"]
        forecast_data["generated_for"] = current_user["email"]
        forecast_data["generated_at"] = datetime.now().isoformat()
        
        return forecast_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@router.post("/forecast/advice")
async def get_forecast_advice(forecast_data: dict, current_user: Dict = Depends(auth_service.get_current_user)):
    """Get AI advice for forecast"""
    try:
        # Add user context
        user_context = f"Customer: {current_user['name']}, Account: {current_user['account_type']}"
        
        # Generate advice
        advice = llm_advisor.generate_weekly_advice(forecast_data)
        
        return {
            "user_id": current_user["user_id"],
            "advice": advice,
            "context": user_context,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advice generation failed: {str(e)}")