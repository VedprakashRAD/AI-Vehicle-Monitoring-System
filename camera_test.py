#!/usr/bin/env python3
"""
Simple camera test to verify video functionality
"""

import cv2
import numpy as np
import time

def test_camera():
    """Test camera functionality"""
    print("🔍 Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Camera resolution: {width}x{height}")
    print(f"🎬 Camera FPS: {fps}")
    
    # Test frame capture
    print("📸 Testing frame capture...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame {i+1}: {frame.shape}")
            
            # Add some text to the frame
            cv2.putText(frame, f"Test Frame {i+1}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save test frame
            cv2.imwrite(f'test_frame_{i+1}.jpg', frame)
        else:
            print(f"❌ Failed to capture frame {i+1}")
        
        time.sleep(0.5)
    
    cap.release()
    print("🎉 Camera test completed successfully!")
    return True

def create_test_frame():
    """Create a test frame for debugging"""
    print("🎨 Creating test frame...")
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (100, 150, 200)  # Light blue background
    
    # Add test elements
    cv2.rectangle(frame, (50, 50), (590, 430), (255, 255, 255), 2)
    cv2.putText(frame, "AI Vehicle Monitoring System", (120, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Camera Test Frame", (200, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "If you see this, video is working!", (150, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add some shapes
    cv2.circle(frame, (320, 300), 50, (0, 255, 0), 2)
    cv2.rectangle(frame, (250, 350), (390, 400), (0, 0, 255), 2)
    
    # Save test frame
    cv2.imwrite('test_display_frame.jpg', frame)
    print("✅ Test frame saved as 'test_display_frame.jpg'")
    
    return frame

if __name__ == "__main__":
    print("🚗 Vehicle Monitoring System - Camera Test")
    print("=" * 50)
    
    # Test camera
    camera_ok = test_camera()
    
    # Create test frame
    test_frame = create_test_frame()
    
    if camera_ok:
        print("✅ Camera is working properly!")
        print("💡 If the web dashboard video isn't showing:")
        print("   1. Make sure you granted camera permissions to Terminal/Python")
        print("   2. Check if another app is using the camera")
        print("   3. Try refreshing the web page")
        print("   4. Check the browser console for errors")
    else:
        print("❌ Camera issues detected")
        print("💡 Try these troubleshooting steps:")
        print("   1. Check camera permissions in System Settings")
        print("   2. Close other apps that might be using the camera")
        print("   3. Try a different camera source in the web interface")
    
    print(f"\n📸 Test images saved in: {__file__.replace('camera_test.py', '')}")
    print("🌐 Web dashboard should be at: http://localhost:8080")
