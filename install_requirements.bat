@echo off
echo Installing Python dependencies...
echo.

cd server

pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Dependencies installed successfully!
) else (
    echo.
    echo ❌ Failed to install dependencies
)

pause