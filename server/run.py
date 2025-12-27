import subprocess
import sys
import os
from threading import Thread
import time

def run_api():
    """Run the main API"""
    print("🚀 Starting Legacy API...")
    os.system(f'"{sys.executable}" api.py')

def run_enterprise_api():
    """Run the enterprise API"""
    print("🚀 Starting Enterprise API...")
    os.system(f'"{sys.executable}" enterprise_api.py')

if __name__ == "__main__":
    print("=" * 70)
    print("🔌 Energy Management System - Unified Launcher")
    print("=" * 70)
    print("1. Legacy API: http://localhost:8000")
    print("2. Enterprise API: http://localhost:8001")
    print("3. Legacy Frontend: client/index.html")
    print("4. Enterprise Frontend: client/login.html")
    print("=" * 70)
    
    choice = input("\nSelect mode:\n1. Legacy only\n2. Enterprise only\n3. Both\nYour choice (1-3): ")
    
    if choice == "1":
        run_api()
    elif choice == "2":
        # Modify enterprise API to run on port 8001
        os.environ["API_PORT"] = "8001"
        run_enterprise_api()
    elif choice == "3":
        # Run both on different ports
        os.environ["API_PORT"] = "8001"
        
        # Start enterprise API in background thread
        enterprise_thread = Thread(target=run_enterprise_api)
        enterprise_thread.daemon = True
        enterprise_thread.start()
        
        # Wait a bit then start legacy API
        time.sleep(2)
        run_api()
    else:
        print("Invalid choice. Running legacy API by default.")
        run_api()