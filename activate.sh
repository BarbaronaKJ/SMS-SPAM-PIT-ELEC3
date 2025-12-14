#!/bin/bash
# Quick activation script for the virtual environment

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
fi

echo "Activating virtual environment..."
echo "To deactivate, type: deactivate"
source venv/bin/activate

