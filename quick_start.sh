#!/bin/bash

echo "=========================================="
echo "SMS Spam Classification - Quick Start"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Step 1: Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ ! -f "venv/.deps_installed" ]; then
    echo "Step 2: Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.deps_installed
else
    echo "Dependencies already installed."
fi

echo ""
echo "Step 3: Training models..."
python text_classification.py

echo ""
echo "Step 4: Starting Flask API server..."
echo "The API will be available at http://localhost:5000"
echo "Web GUI will be available at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
python app.py

