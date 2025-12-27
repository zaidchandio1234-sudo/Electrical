#!/usr/bin/env python3
"""
Quick test script to verify API endpoints
Run this while your api.py server is running
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 60)
print("🔍 TESTING ENERGYWISE PRO API ENDPOINTS")
print("=" * 60)

# Test 1: Health check
print("\n1️⃣  Testing /health endpoint...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✅ Health check PASSED")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ Health check FAILED: {response.status_code}")
except Exception as e:
    print(f"❌ Health check ERROR: {e}")

# Test 2: Generate Forecast (POST)
print("\n2️⃣  Testing POST /generate-forecast endpoint...")
try:
    response = requests.post(f"{API_URL}/generate-forecast", timeout=10)
    if response.status_code == 200:
        print("✅ POST /generate-forecast PASSED")
        data = response.json()
        print(f"   Generated {len(data.get('predictions', []))} days of predictions")
        print(f"   Session ID: {data.get('session_id', 'N/A')}")
    else:
        print(f"❌ POST /generate-forecast FAILED: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ POST /generate-forecast ERROR: {e}")

# Test 3: Generate Forecast (GET)
print("\n3️⃣  Testing GET /generate-forecast endpoint...")
try:
    response = requests.get(f"{API_URL}/generate-forecast", timeout=10)
    if response.status_code == 200:
        print("✅ GET /generate-forecast PASSED")
        data = response.json()
        print(f"   Generated {len(data.get('predictions', []))} days of predictions")
    else:
        print(f"❌ GET /generate-forecast FAILED: {response.status_code}")
except Exception as e:
    print(f"❌ GET /generate-forecast ERROR: {e}")

# Test 4: AI Predict
print("\n4️⃣  Testing POST /api/v1/predict endpoint...")
try:
    payload = {
        "session_id": "test_session_123",
        "include_confidence": True,
        "model_type": "ensemble_ai"
    }
    response = requests.post(f"{API_URL}/api/v1/predict", json=payload, timeout=10)
    if response.status_code == 200:
        print("✅ POST /api/v1/predict PASSED")
        data = response.json()
        print(f"   Generated {len(data.get('predictions', []))} predictions")
    else:
        print(f"❌ POST /api/v1/predict FAILED: {response.status_code}")
except Exception as e:
    print(f"❌ POST /api/v1/predict ERROR: {e}")

# Test 5: AI Chat
print("\n5️⃣  Testing POST /api/v1/ai/chat endpoint...")
try:
    payload = {
        "question": "How can I save money on electricity?",
        "context": {
            "peak_day": "Thursday",
            "peak_value": 52.1,
            "weekly_total": 312.9
        }
    }
    response = requests.post(f"{API_URL}/api/v1/ai/chat", json=payload, timeout=10)
    if response.status_code == 200:
        print("✅ POST /api/v1/ai/chat PASSED")
        data = response.json()
        print(f"   AI Response: {data.get('response', 'N/A')[:100]}...")
    else:
        print(f"❌ POST /api/v1/ai/chat FAILED: {response.status_code}")
except Exception as e:
    print(f"❌ POST /api/v1/ai/chat ERROR: {e}")

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)
print("\n💡 INSTRUCTIONS:")
print("   If any tests FAILED with 404:")
print("   1. Stop your current API server (Ctrl+C)")
print("   2. Make sure you're using the UPDATED api.py file")
print("   3. Restart: python api.py")
print("   4. Run this test again: python test_api.py")
print("=" * 60)