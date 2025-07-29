#!/usr/bin/env python3
"""
Simple Advanced Vehicle Monitoring Dashboard
"""

from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import logging
import os
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vehicle-monitoring-2025'

# Global variables
monitoring_active = False
current_frame = None
detection_count = 0
model = None

def initialize_model():
    """Initialize YOLO model"""
    global model
    try:
        model = YOLO('yolov8l.pt')
        logger.info("YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return False

def generate_frames():
    """Generate video frames with detection"""
    global monitoring_active, current_frame, detection_count, model
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    while monitoring_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            if model is not None:
                # Run detection
                results = model(frame, verbose=False)
                detection_count = 0
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf.cpu().numpy())
                            if conf < 0.3:
                                continue
                            
                            cls = int(box.cls.cpu().numpy())
                            # Vehicle classes: car=2, motorcycle=3, bus=5, truck=7
                            if cls in [2, 3, 5, 7]:
                                detection_count += 1
                                bbox = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw label
                                vehicle_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                                label = f"{vehicle_names.get(cls, 'vehicle')} {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add timestamp and detection count
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            current_frame = frame
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            continue
    
    cap.release()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('simple_advanced.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring"""
    global monitoring_active
    
    if not monitoring_active:
        monitoring_active = True
        # Start frame generation in background thread
        thread = threading.Thread(target=lambda: list(generate_frames()), daemon=True)
        thread.start()
        return jsonify({'message': 'Monitoring started successfully'})
    else:
        return jsonify({'message': 'Monitoring already active'})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    global monitoring_active
    monitoring_active = False
    return jsonify({'message': 'Monitoring stopped successfully'})

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    return jsonify({
        'total_vehicles_today': detection_count,
        'active_cameras': 1 if monitoring_active else 0,
        'violations_today': 0,
        'average_speed': 0,
        'monitoring_active': monitoring_active
    })

if __name__ == '__main__':
    if initialize_model():
        logger.info("Starting Advanced Vehicle Monitoring Dashboard")
        logger.info("Dashboard available at: http://localhost:8080")
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    else:
        logger.error("Failed to initialize model")