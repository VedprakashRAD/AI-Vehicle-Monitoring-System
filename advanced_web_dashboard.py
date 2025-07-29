#!/usr/bin/env python3
"""
Advanced Web Dashboard for Vehicle Monitoring System
==================================================

Features:
- Multi-camera live feeds
- Real-time analytics dashboard
- Interactive traffic heatmaps
- Vehicle detection statistics
- Traffic violation alerts
- System performance monitoring
- Historical data visualization
- RESTful API endpoints
- WebSocket real-time updates
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
from flask_restx import Api, Resource, fields, Namespace
import cv2
import numpy as np
import json
import threading
import time
import base64
from datetime import datetime, timedelta
from collections import defaultdict
import sqlite3
import logging
import os
from advanced_system import (
    AdvancedVehicleMonitoringSystem, 
    CameraConfig, 
    VehicleDetection,
    SystemMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'advanced-vehicle-monitoring-2025'

# Initialize Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Flask-RESTX API
api = Api(app, version='2.0', title='Advanced Vehicle Monitoring API',
          description='Comprehensive API for AI-powered vehicle monitoring system',
          doc='/api/docs/', prefix='/api')

# Create API namespaces
cameras_ns = Namespace('cameras', description='Camera management operations')
analytics_ns = Namespace('analytics', description='Analytics and reporting operations')
system_ns = Namespace('system', description='System monitoring operations')
alerts_ns = Namespace('alerts', description='Alert and notification operations')

api.add_namespace(cameras_ns, path='/cameras')
api.add_namespace(analytics_ns, path='/analytics')
api.add_namespace(system_ns, path='/system')
api.add_namespace(alerts_ns, path='/alerts')

# Global variables
monitoring_system = None
frame_generators = {}
latest_frames = {}
system_stats = {
    'total_vehicles_today': 0,
    'active_cameras': 0,
    'violations_today': 0,
    'average_speed': 0,
    'peak_hour_traffic': 0,
    'camera_stats': {}
}

# API Models for documentation
vehicle_model = api.model('Vehicle', {
    'id': fields.Integer(required=True, description='Vehicle ID'),
    'class_name': fields.String(required=True, description='Vehicle type'),
    'confidence': fields.Float(required=True, description='Detection confidence'),
    'speed': fields.Float(description='Vehicle speed in km/h'),
    'license_plate': fields.String(description='License plate number'),
    'color': fields.String(description='Vehicle color'),
    'timestamp': fields.DateTime(required=True, description='Detection timestamp')
})

camera_model = api.model('Camera', {
    'id': fields.String(required=True, description='Camera ID'),
    'name': fields.String(required=True, description='Camera name'),
    'status': fields.String(required=True, description='Camera status'),
    'active_tracks': fields.Integer(description='Number of active tracks'),
    'total_detections': fields.Integer(description='Total detections')
})

system_metrics_model = api.model('SystemMetrics', {
    'cpu_usage': fields.Float(description='CPU usage percentage'),
    'memory_usage': fields.Float(description='Memory usage percentage'),
    'gpu_usage': fields.Float(description='GPU usage percentage'),
    'processing_fps': fields.Float(description='Processing frames per second'),
    'active_tracks': fields.Integer(description='Total active tracks'),
    'timestamp': fields.DateTime(description='Metrics timestamp')
})

def initialize_monitoring_system():
    """Initialize the advanced monitoring system"""
    global monitoring_system
    
    try:
        monitoring_system = AdvancedVehicleMonitoringSystem('advanced_config.yaml')
        
        # Add default camera if none configured
        demo_camera = CameraConfig(
            id='main_camera',
            name='Main Entrance Camera',
            source=0,  # Default webcam
            resolution=(1920, 1080),
            fps=30
        )
        monitoring_system.add_camera(demo_camera)
        
        logger.info("Advanced monitoring system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring system: {e}")
        return False

def get_database_stats():
    """Get statistics from the database"""
    try:
        if not os.path.exists('advanced_vehicle_monitoring.db'):
            return system_stats
            
        conn = sqlite3.connect('advanced_vehicle_monitoring.db')
        cursor = conn.cursor()
        
        # Get today's vehicle count
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT COUNT(*) FROM detections 
            WHERE DATE(timestamp) = ?
        ''', (today,))
        system_stats['total_vehicles_today'] = cursor.fetchone()[0]
        
        # Get violations count
        cursor.execute('''
            SELECT COUNT(*) FROM detections 
            WHERE DATE(timestamp) = ? AND violations IS NOT NULL
        ''', (today,))
        system_stats['violations_today'] = cursor.fetchone()[0]
        
        # Get average speed
        cursor.execute('''
            SELECT AVG(speed) FROM detections 
            WHERE DATE(timestamp) = ? AND speed IS NOT NULL
        ''', (today,))
        avg_speed = cursor.fetchone()[0]
        system_stats['average_speed'] = round(avg_speed, 1) if avg_speed else 0
        
        # Get peak hour traffic
        cursor.execute('''
            SELECT MAX(hourly_count) FROM (
                SELECT COUNT(*) as hourly_count 
                FROM detections 
                WHERE DATE(timestamp) = ?
                GROUP BY strftime('%H', timestamp)
            )
        ''', (today,))
        peak_traffic = cursor.fetchone()[0]
        system_stats['peak_hour_traffic'] = peak_traffic if peak_traffic else 0
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
    
    return system_stats

def generate_annotated_frame(camera_id):
    """Generate annotated frames for a camera"""
    global monitoring_system, latest_frames
    
    if monitoring_system is None:
        return
    
    # Get camera configuration
    camera_config = monitoring_system.cameras.get(camera_id)
    if not camera_config:
        return
    
    cap = cv2.VideoCapture(camera_config.source)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.resolution[1])
    cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # Run detection
            detections = monitoring_system._detect_vehicles(frame, camera_config)
            
            # Draw detections on frame
            annotated_frame = draw_detections(frame, detections)
            
            # Store latest frame
            latest_frames[camera_id] = annotated_frame
            
            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {e}")
            continue
    
    cap.release()

def draw_detections(frame, detections):
    """Draw detection annotations on frame"""
    annotated_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        
        # Choose color based on vehicle type
        color_map = {
            'car': (0, 255, 0),
            'motorcycle': (255, 0, 0),
            'bus': (0, 0, 255),
            'truck': (255, 255, 0)
        }
        color = color_map.get(detection.class_name, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label_parts = [f"{detection.class_name} ({detection.confidence:.2f})"]
        if detection.speed:
            label_parts.append(f"{detection.speed:.1f} km/h")
        if detection.license_plate:
            label_parts.append(detection.license_plate)
        
        label = " | ".join(label_parts)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw detection count
    detection_count = f"Detections: {len(detections)}"
    cv2.putText(annotated_frame, detection_count, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame

# Web Routes
@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        return render_template('advanced_dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        # Return improved inline HTML dashboard
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>Advanced Vehicle Monitoring System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); backdrop-filter: blur(10px); }
        .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .status { padding: 15px; color: white; border-radius: 10px; margin: 20px 0; text-align: center; font-weight: bold; box-shadow: 0 4px 15px rgba(39,174,96,0.3); transition: all 0.3s ease; }
        .status.running { background: linear-gradient(45deg, #27ae60, #2ecc71); }
        .status.stopped { background: linear-gradient(45deg, #e74c3c, #c0392b); }
        .features { list-style: none; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .features li { padding: 15px; background: linear-gradient(45deg, #f8f9fa, #e9ecef); border-radius: 10px; border-left: 4px solid #3498db; transition: transform 0.3s ease; }
        .features li:hover { transform: translateX(5px); }
        .btn { display: inline-block; padding: 12px 25px; background: linear-gradient(45deg, #3498db, #2980b9); color: white; text-decoration: none; border-radius: 25px; margin: 10px; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(52,152,219,0.3); }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(52,152,219,0.4); }
        .video-container { text-align: center; margin: 30px 0; background: #000; border-radius: 15px; padding: 20px; position: relative; overflow: hidden; }
        .video-feed { max-width: 100%; height: auto; border-radius: 10px; display: block; margin: 0 auto; background: #000; min-height: 400px; }
        .video-status { position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.7); color: white; padding: 8px 12px; border-radius: 20px; font-size: 0.9rem; }
        .video-status.live { background: rgba(231,76,60,0.8); }
        .video-status.offline { background: rgba(149,165,166,0.8); }
        .controls { text-align: center; margin: 20px 0; }
        .control-btn { background: linear-gradient(45deg, #e74c3c, #c0392b); color: white; border: none; padding: 12px 24px; border-radius: 25px; margin: 0 10px; cursor: pointer; font-weight: bold; transition: all 0.3s ease; font-size: 1rem; }
        .control-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(0,0,0,0.3); }
        .control-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .control-btn.start { background: linear-gradient(45deg, #27ae60, #2ecc71); }
        .control-btn.start:hover { box-shadow: 0 6px 15px rgba(39,174,96,0.4); }
        .control-btn.stop { background: linear-gradient(45deg, #e74c3c, #c0392b); }
        .control-btn.stop:hover { box-shadow: 0 6px 15px rgba(231,76,60,0.4); }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin: 30px 0; }
        .stat-card { background: linear-gradient(45deg, #fff, #f8f9fa); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 25px rgba(0,0,0,0.1); border-top: 4px solid #3498db; transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px); box-shadow: 0 12px 35px rgba(0,0,0,0.15); }
        .stat-number { font-size: 2.5rem; font-weight: bold; color: #2c3e50; margin-bottom: 8px; }
        .stat-label { color: #7f8c8d; margin-top: 5px; font-size: 0.95rem; }
        .footer { text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 0.9rem; }
        .loading { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,.3); border-radius: 50%; border-top-color: #fff; animation: spin 1s ease-in-out infinite; }
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .notification { position: fixed; top: 20px; right: 20px; padding: 15px 20px; border-radius: 8px; color: white; font-weight: bold; z-index: 1000; transition: all 0.3s ease; }
        .notification.success { background: linear-gradient(45deg, #27ae60, #2ecc71); }
        .notification.error { background: linear-gradient(45deg, #e74c3c, #c0392b); }
        .notification.warning { background: linear-gradient(45deg, #f39c12, #e67e22); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 Advanced Vehicle Monitoring System</h1>
            <p>AI-Powered Real-Time Traffic Analysis & Detection</p>
        </div>
        
        <div id="systemStatus" class="status running">
            ✅ System is ready! YOLOv8 AI model loaded and ready.
        </div>
        
        <div class="controls">
            <button id="startBtn" class="control-btn start" onclick="startMonitoring()">▶️ Start Monitoring</button>
            <button id="stopBtn" class="control-btn stop" onclick="stopMonitoring()" disabled>⏹️ Stop Monitoring</button>
        </div>
        
        <div class="video-container">
            <div id="videoStatus" class="video-status offline">⚫ OFFLINE</div>
            <h3 style="color: white; margin-bottom: 20px;">Live Camera Feed with AI Detection</h3>
            <img id="videoFeed" class="video-feed" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='400'%3E%3Crect width='100%25' height='100%25' fill='%23000'/%3E%3Ctext x='50%25' y='50%25' font-size='24' fill='%23666' text-anchor='middle' dy='.3em'%3ECamera feed will appear here when monitoring starts%3C/text%3E%3C/svg%3E" alt="Live Camera Feed" onerror="handleVideoError()">
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="vehicleCount">-</div>
                <div class="stat-label">Vehicles Detected Today</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="cameraCount">-</div>
                <div class="stat-label">Active Cameras</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="violationCount">-</div>
                <div class="stat-label">Traffic Violations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="avgSpeed">-</div>
                <div class="stat-label">Average Speed (km/h)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="peakTraffic">-</div>
                <div class="stat-label">Peak Hour Traffic</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="systemUptime">-</div>
                <div class="stat-label">System Uptime</div>
            </div>
        </div>
        
        <h3>🌟 Available Features:</h3>
        <ul class="features">
            <li>📹 <strong>Real-time Vehicle Detection</strong><br>Advanced YOLOv8 AI model for accurate vehicle identification</li>
            <li>🚗 <strong>Multi-Vehicle Classification</strong><br>Detects cars, motorcycles, buses, trucks with confidence scores</li>
            <li>📊 <strong>Live Traffic Analytics</strong><br>Real-time statistics and traffic flow analysis</li>
            <li>🗺️ <strong>License Plate Recognition</strong><br>Automatic number plate detection and OCR processing</li>
            <li>⚡ <strong>Speed Estimation</strong><br>Calculate vehicle speeds and detect violations</li>
            <li>📈 <strong>Historical Data</strong><br>Store and analyze traffic patterns over time</li>
        </ul>
        
        <div style="text-align: center; margin-top: 30px;">
            <a href="/api/docs/" class="btn" target="_blank">📚 API Documentation</a>
            <a href="/analytics" class="btn">📊 Analytics Dashboard</a>
        </div>
        
        <div class="footer">
            <p><strong>System started at:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Dashboard URL:</strong> http://localhost:8080</p>
            <p>🔬 Powered by YOLOv8, OpenCV, and Flask | 🚀 Real-time AI Processing</p>
        </div>
    </div>
    
    <script>
        let monitoringActive = false;
        let statsUpdateInterval;
        let systemStartTime = new Date();
        
        // Show notification function
        function showNotification(message, type) {
            type = type || "success";
            var notification = document.createElement('div');
            notification.className = 'notification ' + type;
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(function() {
                notification.remove();
            }, 4000);
        }
        
        // Start monitoring function
        function startMonitoring() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const systemStatus = document.getElementById('systemStatus');
            const videoStatus = document.getElementById('videoStatus');
            const videoFeed = document.getElementById('videoFeed');
            
            startBtn.disabled = true;
            startBtn.innerHTML = '<div class="loading"></div> Starting...';
            
            fetch('/api/cameras/main_camera/start', { method: 'POST' })
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.message) {
                        monitoringActive = true;
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        startBtn.innerHTML = '▶️ Start Monitoring';
                        
                        systemStatus.className = 'status running';
                        systemStatus.innerHTML = '🟢 Monitoring ACTIVE - AI detection in progress';
                        
                        videoStatus.className = 'video-status live';
                        videoStatus.innerHTML = '🔴 LIVE';
                        
                        videoFeed.src = '/video_feed/main_camera?' + new Date().getTime();
                        
                        // Start stats updates
                        startStatsUpdates();
                        
                        showNotification('✅ Monitoring started successfully!', 'success');
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                })
                .catch(function(error) {
                    startBtn.disabled = false;
                    startBtn.innerHTML = '▶️ Start Monitoring';
                    showNotification('❌ Failed to start monitoring: ' + error.message, 'error');
                    console.error('Start monitoring error:', error);
                });
        }
        
        // Stop monitoring function
        function stopMonitoring() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const systemStatus = document.getElementById('systemStatus');
            const videoStatus = document.getElementById('videoStatus');
            const videoFeed = document.getElementById('videoFeed');
            
            stopBtn.disabled = true;
            stopBtn.innerHTML = '<div class="loading"></div> Stopping...';
            
            fetch('/api/cameras/main_camera/stop', { method: 'POST' })
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.message) {
                        monitoringActive = false;
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                        stopBtn.innerHTML = '⏹️ Stop Monitoring';
                        
                        systemStatus.className = 'status stopped';
                        systemStatus.innerHTML = '⏸️ Monitoring STOPPED - System ready to start';
                        
                        videoStatus.className = 'video-status offline';
                        videoStatus.innerHTML = '⚫ OFFLINE';
                        
                        videoFeed.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='400'%3E%3Crect width='100%25' height='100%25' fill='%23000'/%3E%3Ctext x='50%25' y='50%25' font-size='24' fill='%23666' text-anchor='middle' dy='.3em'%3ECamera feed stopped%3C/text%3E%3C/svg%3E";
                        
                        // Stop stats updates
                        stopStatsUpdates();
                        
                        showNotification('⏹️ Monitoring stopped successfully!', 'success');
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }
                })
                .catch(function(error) {
                    stopBtn.disabled = false;
                    stopBtn.innerHTML = '⏹️ Stop Monitoring';
                    showNotification('❌ Failed to stop monitoring: ' + error.message, 'error');
                    console.error('Stop monitoring error:', error);
                });
        }
        
        // Auto-refresh video feed if it fails to load
        function handleVideoError() {
            if (monitoringActive) {
                console.log('Video feed error, attempting to reconnect...');
                setTimeout(function() {
                    document.getElementById('videoFeed').src = '/video_feed/main_camera?' + new Date().getTime();
                }, 3000);
            }
        }
        
        // Update system uptime
        function updateUptime() {
            const now = new Date();
            const uptimeMs = now - systemStartTime;
            const hours = Math.floor(uptimeMs / (1000 * 60 * 60));
            const minutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((uptimeMs % (1000 * 60)) / 1000);
            
            document.getElementById('systemUptime').textContent = (hours < 10 ? '0' + hours : hours) + ':' + (minutes < 10 ? '0' + minutes : minutes) + ':' + (seconds < 10 ? '0' + seconds : seconds);
        }
        
        // Update stats function
        function updateStats() {
            fetch('/api/analytics/stats')
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    document.getElementById('vehicleCount').textContent = data.total_vehicles_today || 0;
                    document.getElementById('cameraCount').textContent = data.active_cameras || 0;
                    document.getElementById('violationCount').textContent = data.violations_today || 0;
                    document.getElementById('avgSpeed').textContent = data.average_speed ? data.average_speed.toFixed(1) : 0;
                    document.getElementById('peakTraffic').textContent = data.peak_hour_traffic || 0;
                })
                .catch(function(error) {
                    console.log('Stats update error:', error);
                    // Don't show error notification for stats updates to avoid spam
                });
        }
        
        // Start stats updates
        function startStatsUpdates() {
            updateStats(); // Initial update
            statsUpdateInterval = setInterval(updateStats, 5000);
        }
        
        // Stop stats updates
        function stopStatsUpdates() {
            if (statsUpdateInterval) {
                clearInterval(statsUpdateInterval);
                statsUpdateInterval = null;
            }
        }
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            // Update uptime every second
            setInterval(updateUptime, 1000);
            
            // Initial stats load (even when not monitoring)
            updateStats();
        });
    </script>
</body>
</html>'''

@app.route('/camera/<camera_id>')
def camera_view(camera_id):
    """Individual camera view"""
    return render_template('camera_view.html', camera_id=camera_id)

@app.route('/analytics')
def analytics_page():
    """Analytics and reporting page"""
    return render_template('analytics.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Video streaming endpoint"""
    if camera_id not in frame_generators:
        frame_generators[camera_id] = generate_annotated_frame(camera_id)
    
    return Response(frame_generators[camera_id],
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# REST API Endpoints

@cameras_ns.route('/')
class CameraList(Resource):
    @cameras_ns.marshal_list_with(camera_model)
    def get(self):
        """Get list of all cameras"""
        if monitoring_system is None:
            return []
        
        status = monitoring_system.get_system_status()
        cameras = []
        
        for camera_id, camera_info in status['cameras'].items():
            cameras.append({
                'id': camera_id,
                'name': camera_info['name'],
                'status': camera_info['status'],
                'active_tracks': camera_info['active_tracks'],
                'total_detections': 0  # Would be calculated from database
            })
        
        return cameras

@cameras_ns.route('/<string:camera_id>/start')
class StartCamera(Resource):
    def post(self, camera_id):
        """Start monitoring for a specific camera"""
        if monitoring_system is None:
            return {'error': 'Monitoring system not initialized'}, 500
        
        try:
            if not monitoring_system.is_running:
                monitoring_system.start_monitoring()
            
            return {'message': f'Camera {camera_id} monitoring started'}, 200
        except Exception as e:
            return {'error': str(e)}, 500

@cameras_ns.route('/<string:camera_id>/stop')
class StopCamera(Resource):
    def post(self, camera_id):
        """Stop monitoring for a specific camera"""
        if monitoring_system is None:
            return {'error': 'Monitoring system not initialized'}, 500
        
        try:
            monitoring_system.stop_monitoring()
            return {'message': f'Camera {camera_id} monitoring stopped'}, 200
        except Exception as e:
            return {'error': str(e)}, 500

@analytics_ns.route('/stats')
class AnalyticsStats(Resource):
    def get(self):
        """Get comprehensive analytics statistics"""
        stats = get_database_stats()
        
        if monitoring_system:
            system_status = monitoring_system.get_system_status()
            stats['active_cameras'] = len([c for c in system_status['cameras'].values() 
                                         if c['status'] == 'active'])
        
        return stats

@analytics_ns.route('/heatmap/<string:camera_id>')
class TrafficHeatmap(Resource):
    def get(self, camera_id):
        """Get traffic density heatmap for a camera"""
        if monitoring_system is None:
            return {'error': 'Monitoring system not initialized'}, 500
        
        try:
            # Get time range from query parameters
            hours = request.args.get('hours', 24, type=int)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Generate heatmap data
            heatmap_data = monitoring_system.analytics.generate_heatmap(
                camera_id, (start_time, end_time)
            )
            
            # Convert numpy array to list for JSON serialization
            return {
                'camera_id': camera_id,
                'heatmap': heatmap_data.tolist(),
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                }
            }
        except Exception as e:
            return {'error': str(e)}, 500

@system_ns.route('/status')
class SystemStatus(Resource):
    @system_ns.marshal_with(system_metrics_model)
    def get(self):
        """Get system status and performance metrics"""
        if monitoring_system is None:
            return {'error': 'Monitoring system not initialized'}, 500
        
        status = monitoring_system.get_system_status()
        return status['metrics']

@system_ns.route('/health')
class SystemHealth(Resource):
    def get(self):
        """Get system health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'monitoring_system': monitoring_system is not None,
                'database': os.path.exists('advanced_vehicle_monitoring.db'),
                'api': True
            }
        }
        
        # Check if all services are running
        all_healthy = all(health_status['services'].values())
        health_status['status'] = 'healthy' if all_healthy else 'degraded'
        
        return health_status

@alerts_ns.route('/')
class AlertsList(Resource):
    def get(self):
        """Get list of recent alerts"""
        try:
            conn = sqlite3.connect('advanced_vehicle_monitoring.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT alert_type, camera_id, message, severity, timestamp
                FROM alerts 
                WHERE acknowledged = FALSE
                ORDER BY timestamp DESC 
                LIMIT 50
            ''')
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'type': row[0],
                    'camera_id': row[1],
                    'message': row[2],
                    'severity': row[3],
                    'timestamp': row[4]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            return {'error': str(e)}, 500

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected to WebSocket')
    emit('status', {'message': 'Connected to advanced monitoring system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected from WebSocket')

@socketio.on('request_stats')
def handle_stats_request():
    """Handle statistics request"""
    stats = get_database_stats()
    emit('stats_update', stats)

def broadcast_real_time_updates():
    """Broadcast real-time updates to connected clients"""
    while True:
        try:
            if monitoring_system and monitoring_system.is_running:
                # Get current statistics
                stats = get_database_stats()
                
                # Get system metrics
                system_status = monitoring_system.get_system_status()
                
                # Broadcast to all connected clients
                socketio.emit('stats_update', stats)
                socketio.emit('system_metrics', system_status['metrics'])
            
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")
            time.sleep(10)

if __name__ == '__main__':
    # Initialize the monitoring system
    if initialize_monitoring_system():
        # Start background thread for real-time updates
        update_thread = threading.Thread(target=broadcast_real_time_updates, daemon=True)
        update_thread.start()
        
        logger.info("Starting advanced vehicle monitoring web dashboard")
        socketio.run(app, host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to initialize monitoring system")
