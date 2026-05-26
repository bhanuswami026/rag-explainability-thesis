#!/bin/bash

# RAG Explainability Thesis Project - Startup Script
# Activates the Python virtual environment and launches the Streamlit dashboard.

# Get current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "🎓 Starting RAG Explainability Thesis Interactive Dashboard..."
echo "=========================================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment 'venv' not found."
    echo "Please create it using: python3 -m venv venv"
    echo "And install dependencies: ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "✔ Activating Python virtual environment..."
source venv/bin/activate

# Launch Streamlit app
echo "✔ Starting Streamlit dashboard..."
streamlit run app/main.py
