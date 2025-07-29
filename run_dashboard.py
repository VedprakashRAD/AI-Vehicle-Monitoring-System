#!/usr/bin/env python3
"""
AI-Powered Vehicle Monitoring System - Dashboard Runner
Run this script to start the web dashboard with YOLOv8 integration
"""

import sys
import os
import logging
import argparse

# Add src to path
sys.path.append('src')

def main():
    """Main function to run the dashboard"""
    parser = argparse.ArgumentParser(description='AI-Powered Vehicle Monitoring Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        print("🚗 AI-Powered Vehicle Monitoring System")
        print("=" * 50)
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"Debug: {args.debug}")
        print(f"Confidence Threshold: {args.confidence}")
        print("=" * 50)
        
        # Import and initialize dashboard
        from web.dashboard import VehicleDashboard
        
        dashboard = VehicleDashboard(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
        print("\n🚀 Starting dashboard...")
        print(f"📱 Access the dashboard at: http://{args.host}:{args.port}")
        print("🔄 Press Ctrl+C to stop")
        print("\nFeatures available:")
        print("   • YOLOv8-based AI vehicle detection")
        print("   • Real-time vehicle tracking and counting")
        print("   • Entry/exit logging with timestamps")
        print("   • Speed estimation")
        print("   • Live video feed with annotations")
        print("   • Light/dark theme toggle")
        print("\n" + "=" * 50)
        
        # Run the dashboard
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
        logger.info("Dashboard stopped by user interrupt")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("   pip install flask flask-socketio opencv-python ultralytics")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        logger.error(f"Dashboard startup error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
