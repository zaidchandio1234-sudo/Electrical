#!/bin/bash

echo "🔋 AI Smart Meter Behavioral Advisor - Setup Script"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python installation
echo "Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_success "Python 3 found: $(python3 --version)"

# Check if we're in the right directory
if [ ! -d "server" ] || [ ! -d "client" ]; then
    print_error "Please run this script from the Electrical/ root directory"
    exit 1
fi

# Setup backend
echo ""
echo "📦 Setting up backend..."
cd server

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install requirements
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
print_success "Dependencies installed"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p artifacts
mkdir -p ../dataset
print_success "Directories created"

# Check for required files
echo ""
echo "🔍 Checking for required files..."

FILES_MISSING=0

if [ ! -f "artifacts/gru_model.pth" ]; then
    print_warning "artifacts/gru_model.pth not found - you need to train your model first"
    FILES_MISSING=1
else
    print_success "Model file found"
fi

if [ ! -f "artifacts/gru_scaler.pkl" ]; then
    print_warning "artifacts/gru_scaler.pkl not found - you need to train your model first"
    FILES_MISSING=1
else
    print_success "Scaler file found"
fi

if [ ! -f "../dataset/electricity_usage.csv" ]; then
    print_warning "dataset/electricity_usage.csv not found - you need to add your dataset"
    FILES_MISSING=1
else
    print_success "Dataset found"
fi

# Check for API token
echo ""
if [ -z "$HF_API_TOKEN" ]; then
    print_warning "HF_API_TOKEN environment variable not set"
    echo "   LLM features will use fallback mode (rule-based advice)"
    echo "   To enable LLM: export HF_API_TOKEN='your_token'"
else
    print_success "HF_API_TOKEN is set"
fi

# Summary
echo ""
echo "=================================================="
if [ $FILES_MISSING -eq 1 ]; then
    print_warning "Setup incomplete - some files are missing"
    echo ""
    echo "Next steps:"
    echo "1. Train your model using train.py to generate:"
    echo "   - server/artifacts/gru_model.pth"
    echo "   - server/artifacts/gru_scaler.pkl"
    echo ""
    echo "2. Add your dataset to:"
    echo "   - dataset/electricity_usage.csv"
    echo ""
    echo "3. (Optional) Set HF_API_TOKEN for LLM features:"
    echo "   export HF_API_TOKEN='your_token'"
else
    print_success "Setup complete! All files are in place."
    echo ""
    echo "🚀 To start the application:"
    echo ""
    echo "Terminal 1 (Backend):"
    echo "  cd server"
    echo "  source venv/bin/activate"
    echo "  python api.py"
    echo ""
    echo "Terminal 2 (Frontend):"
    echo "  cd client"
    echo "  python -m http.server 3000"
    echo ""
    echo "Then open: http://localhost:3000"
fi
echo "=================================================="

cd ..