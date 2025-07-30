#!/usr/bin/env python3
"""
Create a test video for camera fallback when no camera is available
"""

import cv2
import numpy as np
import os

def create_test_video():
    """Create a simple test video with moving objects"""
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Failed to create video writer")
        return False
    
    print(f"🎬 Creating test video: {width}x{height} @ {fps}fps for {duration}s")
    
    try:
        for frame_num in range(total_frames):
            # Create blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (40, 40, 40)  # Dark gray background
            
            # Add title
            cv2.putText(frame, "AI Vehicle Monitoring - Test Video", 
                       (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", 
                       (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add moving rectangle (simulates a vehicle)
            rect_width, rect_height = 80, 40
            x = int((frame_num * 3) % (width + rect_width)) - rect_width
            y = height // 2 - rect_height // 2
            
            if 0 <= x <= width - rect_width:
                cv2.rectangle(frame, (x, y), (x + rect_width, y + rect_height), (0, 255, 0), -1)
                cv2.putText(frame, "Car", (x + 10, y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add second moving rectangle
            rect2_x = int((frame_num * 2 + 100) % (width + rect_width)) - rect_width
            rect2_y = height // 2 + 60
            
            if 0 <= rect2_x <= width - rect_width:
                cv2.rectangle(frame, (rect2_x, rect2_y), 
                             (rect2_x + rect_width, rect2_y + rect_height), (0, 0, 255), -1)
                cv2.putText(frame, "Bus", (rect2_x + 10, rect2_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add counting line
            line_y = height // 2
            cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
            cv2.putText(frame, "COUNTING LINE", (10, line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Show progress
            if frame_num % (fps * 2) == 0:  # Every 2 seconds
                progress = (frame_num / total_frames) * 100
                print(f"📊 Progress: {progress:.1f}%")
    
    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return False
    
    finally:
        out.release()
    
    print(f"✅ Test video created successfully: test_video.mp4")
    return True

if __name__ == "__main__":
    if create_test_video():
        print("🎉 Test video ready for camera fallback!")
    else:
        print("❌ Failed to create test video")
