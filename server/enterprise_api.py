from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # type: ignore
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import routers
from api_extended import router as api_router
from api import app as base_app

# Create enterprise app
app = FastAPI(
    title="EnergyWise Pro Enterprise API",
    description="Enterprise energy management platform with AI-powered insights",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router)

# Mount base API under /v1
from api import app as v1_app
app.mount("/v1", v1_app)

@app.get("/")
async def root():
    return {
        "message": "EnergyWise Pro Enterprise API",
        "version": "2.0.0",
        "services": {
            "authentication": "/api/v1/auth",
            "dashboard": "/api/v1/dashboard",
            "billing": "/api/v1/billing",
            "analytics": "/api/v1/analytics",
            "forecast": "/api/v1/forecast",
            "devices": "/api/v1/devices",
            "alerts": "/api/v1/alerts"
        },
        "documentation": "/api/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "authentication": "active",
            "forecast": "active",
            "billing": "active",
            "analytics": "active"
        }
    }

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 EnergyWise Pro Enterprise Platform")
    print("=" * 70)
    print("🔐 Authentication: Enabled")
    print("📊 Analytics: Enabled")
    print("💰 Billing System: Enabled")
    print("🤖 AI Forecasting: Enabled")
    print("🔗 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/api/docs")
    print("=" * 70)
    
    uvicorn.run(
        "enterprise_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )