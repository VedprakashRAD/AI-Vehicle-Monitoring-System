#!/usr/bin/env python3
"""
Simple run script for AI Vehicle Monitoring System
"""

print("🚗 Starting AI Vehicle Monitoring System...")
print("=" * 50)

try:
    # Import required modules
    from web_dashboard import app, socketio, create_templates, logger
    
    # Create templates directory and files
    create_templates()
    
    print("✅ System initialized successfully!")
    print("🌐 Starting web dashboard...")
    print("📍 URL: http://localhost:8080")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    # Run the application
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    
except KeyboardInterrupt:
    print("\n\n👋 System stopped by user")
except ImportError as e:
    print(f"❌ Missing module: {e}")
    print("💡 Try running: pip install flask flask-socketio opencv-python ultralytics")
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Check that all dependencies are installed correctly")
