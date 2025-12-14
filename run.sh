#!/bin/bash
# Run the Flask app using the virtual environment

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and run the app
source venv/bin/activate
python app.py

