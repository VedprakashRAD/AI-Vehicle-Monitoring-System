#!/usr/bin/env python3
"""
Quick Start Script for AI-Powered Vehicle Monitoring System
"""

import sys
import os
import logging

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Quick start function"""
    try:
        print("🚗 AI-Powered Vehicle Monitoring System - Quick Start")
        print("=" * 60)
        
        # Import dashboard
        from web.dashboard import VehicleDashboard
        
        # Create and run dashboard
        dashboard = VehicleDashboard(
            host='127.0.0.1',
            port=8080,
            debug=False
        )
        
        print("🚀 Starting dashboard on http://127.0.0.1:8080")
        print("🔄 Press Ctrl+C to stop")
        print("\nFeatures:")
        print("   • AI vehicle detection with YOLOv8")
        print("   • Real-time counting and tracking")
        print("   • Entry/exit logging")
        print("   • Live video feed")
        print("   • Web dashboard interface")
        print("=" * 60)
        
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
