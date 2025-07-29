#!/bin/bash

# Check python installation
command -v python3 &>/dev/null || { echo >&2 "Python 3 is required but not installed. Aborting."; exit 1; }

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download pretrained YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "Installation completed successfully."
echo "Activate the virtual environment with 'source venv/bin/activate' (Linux/macOS) or 'venv\\Scripts\\activate' (Windows) and start the application by running 'python web_dashboard.py'."
