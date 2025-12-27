#!/usr/bin/env python3
"""
Startup script to verify and run the correct API
"""

import sys
import os

print("=" * 80)
print("🔍 VERIFYING API FILE...")
print("=" * 80)

# Check if api.py exists
if not os.path.exists('api.py'):
    print("❌ ERROR: api.py not found in current directory!")
    print("   Please make sure api.py is in the same folder as this script.")
    sys.exit(1)

print("✅ Found api.py")

# Check for the generate-forecast endpoint
print("\n🔍 Checking for /generate-forecast endpoint...")
with open('api.py', 'r') as f:
    content = f.read()
    if '@app.post("/generate-forecast")' in content:
        print("✅ Found POST /generate-forecast endpoint")
    else:
        print("❌ WARNING: POST /generate-forecast endpoint NOT found!")
        print("   You may be using an old version of api.py")
        print("   Please download the updated api.py file")
        
    if '@app.get("/generate-forecast")' in content:
        print("✅ Found GET /generate-forecast endpoint")
    else:
        print("❌ WARNING: GET /generate-forecast endpoint NOT found!")

    if '@app.post("/api/v1/ai/chat")' in content:
        print("✅ Found POST /api/v1/ai/chat endpoint")
    else:
        print("❌ WARNING: POST /api/v1/ai/chat endpoint NOT found!")

print("\n" + "=" * 80)
print("🚀 STARTING API SERVER...")
print("=" * 80)

# Import and run
try:
    import uvicorn # type: ignore
    from api import app
    
    print("\n✅ API loaded successfully!")
    print("=" * 80)
    print("📡 Server starting on http://localhost:8000")
    print("📖 Docs available at http://localhost:8000/docs")
    print("=" * 80)
    print("\nEndpoints available:")
    print("  • POST   /generate-forecast")
    print("  • GET    /generate-forecast")
    print("  • POST   /api/v1/predict")
    print("  • POST   /api/v1/ai/chat")
    print("  • GET    /health")
    print("  • GET    /")
    print("=" * 80)
    print("\n⚡ Press Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    
except ImportError as e:
    print(f"\n❌ ERROR: Missing dependencies!")
    print(f"   {e}")
    print("\n💡 Install with: pip install fastapi uvicorn pydantic numpy")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR starting server: {e}")
    sys.exit(1)