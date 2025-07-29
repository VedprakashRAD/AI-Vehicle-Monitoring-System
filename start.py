#!/usr/bin/env python3
"""
Easy startup script for AI Vehicle Monitoring System
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open browser after a short delay"""
    print("🌐 Opening web browser...")
    webbrowser.open('http://localhost:8080')

def main():
    print("🚗 AI Vehicle Monitoring System")
    print("================================")
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in virtual environment")
        print("   Consider running: source venv/bin/activate")
    
    # Check if required files exist
    required_files = ['web_dashboard.py', 'vehicle_counter.py', 'config.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    
    # Test imports
    try:
        import flask
        import ultralytics
        import cv2
        print("✅ Core modules available")
    except ImportError as e:
        print(f"❌ Missing module: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n🚀 Starting web dashboard...")
    print("   URL: http://localhost:8080")
    print("   Press Ctrl+C to stop")
    print("-" * 40)
    
    # Open browser after 3 seconds
    Timer(3.0, open_browser).start()
    
    # Import and run the web dashboard
    try:
        from web_dashboard import app, socketio
        socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error starting web dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
