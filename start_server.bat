@echo off
chcp 65001 >nul
title Sindh House Price Predictor

echo.
echo 🏠 Sindh House Price Predictor - Server
echo ==================================================
echo.

echo 📍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.7+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

echo 🚀 Starting server...
echo.

python main.py

echo.
echo Server stopped.
pause