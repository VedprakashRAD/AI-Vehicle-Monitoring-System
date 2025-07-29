#!/usr/bin/env python3
"""
Test script for the AI-Powered Vehicle Monitoring System
Run this to test the complete system with new YOLOv8 integration
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
    """Main test function"""
    try:
        print("🚗 AI-Powered Vehicle Monitoring System - Test Mode")
        print("=" * 60)
        
        # Test imports
        print("1. Testing imports...")
        from web.dashboard import VehicleDashboard
        from core.vehicle_counter import VehicleCounter, WebVehicleCounter
        from database.manager import DatabaseManager
        print("   ✅ All modules imported successfully")
        
        # Test vehicle counter
        print("\n2. Testing vehicle counter...")
        counter = VehicleCounter(confidence_threshold=0.5, model_path="yolov8n.pt")
        print("   ✅ VehicleCounter initialized")
        
        web_counter = WebVehicleCounter(confidence_threshold=0.5, model_path="yolov8n.pt")
        print("   ✅ WebVehicleCounter initialized")
        
        # Test database
        print("\n3. Testing database...")
        db = DatabaseManager()
        print("   ✅ Database initialized")
        
        # Test dashboard
        print("\n4. Testing dashboard...")
        dashboard = VehicleDashboard(host='127.0.0.1', port=8080, debug=False)
        print("   ✅ Dashboard initialized")
        
        print("\n🎉 All tests passed! System is ready to run.")
        print("\nTo start the web dashboard:")
        print("   python run_dashboard.py")
        print("\nFeatures available:")
        print("   • YOLOv8-based AI vehicle detection")
        print("   • Real-time vehicle tracking and counting")
        print("   • Entry/exit logging with timestamps")
        print("   • Speed estimation")
        print("   • Live video feed with annotations")
        print("   • Web dashboard with statistics")
        print("   • Light/dark theme toggle")
        
        # Show model status
        print(f"\n📊 Model Status:")
        if counter.model is not None:
            print("   • YOLOv8 model: ✅ Loaded")
            print("   • AI Detection: ✅ Active")
        else:
            print("   • YOLOv8 model: ⚠️  Using simulation mode")
            print("   • AI Detection: 🔄 Fallback mode")
        
        print(f"\n🎯 Vehicle Classes Detected:")
        for class_id, vehicle_type in counter.vehicle_classes.items():
            print(f"   • {vehicle_type.title()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ System test completed successfully!")
        
        # Ask if user wants to start the dashboard
        try:
            start_dashboard = input("\nWould you like to start the dashboard now? (y/n): ").lower().strip()
            if start_dashboard in ['y', 'yes']:
                print("\n🚀 Starting dashboard...")
                from web.dashboard import VehicleDashboard
                dashboard = VehicleDashboard(host='127.0.0.1', port=8080, debug=True)
                dashboard.run()
            else:
                print("\n👋 Dashboard not started. Run 'python run_dashboard.py' when ready.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
    else:
        print("\n❌ System test failed. Please check the errors above.")
        sys.exit(1)
