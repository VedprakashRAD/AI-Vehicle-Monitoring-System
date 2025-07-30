#!/usr/bin/env python3
"""
Setup script for improved vehicle detection system
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        'ultralytics>=8.0.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'flask>=2.3.0',
        'flask-socketio>=5.3.0',
        'torch>=1.13.0',
        'torchvision>=0.14.0'
    ]
    
    print("📦 Installing required packages...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {req}")

def download_yolo_models():
    """Download YOLO models"""
    print("🤖 Downloading YOLO models...")
    try:
        from ultralytics import YOLO
        
        models = ['yolov8n.pt', 'yolov8s.pt']
        for model in models:
            try:
                YOLO(model)
                print(f"✅ Downloaded {model}")
            except:
                print(f"❌ Failed to download {model}")
                
    except ImportError:
        print("❌ Ultralytics not available")

def setup_database():
    """Initialize database"""
    print("🗄️ Setting up database...")
    import sqlite3
    
    conn = sqlite3.connect('vehicle_tracking.db')
    cursor = conn.cursor()
    
    # Clear existing data for fresh start
    cursor.execute('DROP TABLE IF EXISTS vehicle_logs')
    
    cursor.execute('''
        CREATE TABLE vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id TEXT,
            registration_number TEXT,
            vehicle_type TEXT,
            status TEXT,
            entry_time DATETIME,
            exit_time DATETIME,
            confidence REAL,
            bbox_x1 INTEGER,
            bbox_y1 INTEGER,
            bbox_x2 INTEGER,
            bbox_y2 INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database setup complete")

def main():
    print("🚗 Setting up AI Vehicle Monitoring System")
    print("=" * 50)
    
    install_requirements()
    download_yolo_models()
    setup_database()
    
    print("\n✅ Setup complete!")
    print("🚀 Run 'python start.py' to start the system")

if __name__ == "__main__":
    main()