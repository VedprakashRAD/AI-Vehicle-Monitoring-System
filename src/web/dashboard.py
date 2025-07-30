"""
Web Dashboard Module
Flask-based web interface for the vehicle monitoring system.
"""

from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import cv2
import threading
import time
import numpy as np
import json
from datetime import datetime
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.vehicle_counter import WebVehicleCounter
from database.manager import DatabaseManager
from utils.system_monitor import system_monitor
from utils.anomaly_detector import anomaly_detector

logger = logging.getLogger(__name__)


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class VehicleDashboard:
    """Main dashboard application class"""
    
    def __init__(self, host='0.0.0.0', port=8080, debug=True, config=None):
        self.host = host
        self.port = port
        self.debug = debug
        self.config = config or {}
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='../../templates',
                        static_folder='../../static')
        self.app.config['SECRET_KEY'] = 'vehicle_monitoring_secret_key'
        
        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize database
        self.db = DatabaseManager()
        
        # Application state
        self.vehicle_counter = None
        self.is_processing = False
        self.camera_source = 0  # Default camera source
        
        # Setup routes and events
        self._setup_routes()
        self._setup_socket_events()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/stats')
        def get_stats():
            if self.vehicle_counter:
                return jsonify(self.vehicle_counter.latest_stats)
            return jsonify({'error': 'No active monitoring'})
        
        @self.app.route('/api/hourly_summary')
        def get_hourly_summary():
            days = int(request.args.get('days', 7))
            data = self.db.get_hourly_summary(days)
            return jsonify(data)
        
        @self.app.route('/api/trend_data')
        def get_trend_data():
            data = self.db.get_trend_data()
            return jsonify(data)
        
        @self.app.route('/api/model_insights')
        def get_model_insights():
            data = self.db.get_model_insights()
            return jsonify(data)
        
        @self.app.route('/api/export_data')
        def export_data():
            format_type = request.args.get('format', 'csv').lower()
            data = self.db.export_data(format_type)
            
            if data is None:
                return jsonify({'error': 'Export failed'}), 500
            
            if format_type == 'csv':
                return Response(data, mimetype='text/csv',
                              headers={'Content-disposition': 'attachment; filename=vehicle_data.csv'})
            elif format_type == 'json':
                return Response(data, mimetype='application/json',
                              headers={'Content-disposition': 'attachment; filename=vehicle_data.json'})
            elif format_type == 'xml':
                return Response(data, mimetype='application/xml',
                              headers={'Content-disposition': 'attachment; filename=vehicle_data.xml'})
            else:
                return jsonify({'error': 'Unsupported format'}), 400
        
        @self.app.route('/api/system_health')
        def get_system_health():
            return jsonify(system_monitor.get_metrics())
        
        @self.app.route('/api/anomalies')
        def get_anomalies():
            return jsonify({
                'active_anomalies': anomaly_detector.get_active_anomalies(),
                'recent_anomalies': anomaly_detector.get_recent_anomalies(10),
                'summary': anomaly_detector.get_anomaly_summary()
            })

        @self.app.route('/api/entry_exit_log')
        def get_entry_exit_log():
            if self.vehicle_counter:
                return jsonify(self.vehicle_counter.get_entry_exit_details())
            return jsonify([])

        @self.app.route('/start_monitoring', methods=['POST'])
        def start_monitoring():
            try:
                data = request.get_json()
                source = data.get('source', 0)
                self.camera_source = source  # Set camera source
                confidence = float(data.get('confidence', 0.5))
                
                self.vehicle_counter = WebVehicleCounter(confidence_threshold=confidence)
                self.is_processing = True
                
                logger.info(f"Started monitoring with source: {source}, confidence: {confidence}")
                return jsonify({'status': 'success', 'message': 'Monitoring started'})
                
            except Exception as e:
                logger.error(f"Error starting monitoring: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/stop_monitoring', methods=['POST'])
        def stop_monitoring():
            self.is_processing = False
            logger.info("Monitoring stopped")
            return jsonify({'status': 'success', 'message': 'Monitoring stopped'})
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def _setup_socket_events(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Client connected')
            emit('status', {'message': 'Connected to vehicle monitoring system'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Client disconnected')
        
        @self.socketio.on('request_stats')
        def handle_stats_request():
            if self.vehicle_counter:
                emit('stats_update', self.vehicle_counter.latest_stats)
    
    def _generate_frames(self):
        """Generate video frames for streaming"""
        logger.info("🔍 Starting video feed generation...")
        
        # Show placeholder until monitoring starts
        while not self.is_processing:
            placeholder = self._create_placeholder_frame()
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)
        
        # Start camera processing
        logger.info("🎬 Opening camera for monitoring...")
        cap = cv2.VideoCapture(self.camera_source if str(self.camera_source).isdigit() else self.camera_source)
        if not cap.isOpened():
            # Try to open test video if camera fails
            cap = cv2.VideoCapture("test_video.mp4")
        
        if not cap.isOpened():
            logger.error("❌ Failed to open camera")
            while self.is_processing:
                error_frame = self._create_error_frame("Camera not accessible")
                _, buffer = cv2.imencode('.jpg', error_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
            return
        
        logger.info("✅ Camera opened successfully")
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame from camera")
                    continue
                
                # Process frame with AI detection
                if self.vehicle_counter is not None:
                    try:
                        processed_frame, stats = self.vehicle_counter.process_frame_for_web(frame)
                        
                        # Update anomaly detector with latest stats
                        anomaly_detector.update_traffic_data(stats)
                        
                        # Add anomaly data to stats
                        stats['anomalies'] = json.loads(json.dumps({
                            'active': anomaly_detector.get_active_anomalies(),
                            'summary': anomaly_detector.get_anomaly_summary()
                        }, default=json_serializer))
                        
                        self.socketio.emit('stats_update', stats)
                    except Exception as e:
                        logger.error(f"❌ Error processing frame: {e}")
                        processed_frame = frame
                else:
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, "AI Detection Active",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            logger.error(f"❌ Error in frame generation: {e}")
        finally:
            cap.release()
            logger.info("📹 Camera released")
    
    def _create_placeholder_frame(self):
        """Create a placeholder frame when no monitoring is active"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        text = "Click 'Start Monitoring' to begin"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 255, 255)
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        return frame
    
    def _create_error_frame(self, error_message):
        """Create an error frame with message"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 80)  # Dark blue background
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (100, 100, 255)
        thickness = 2
        
        # Split long messages into multiple lines
        words = error_message.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) < 35:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        # Draw each line
        y_start = frame.shape[0] // 2 - (len(lines) * 25) // 2
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x = (frame.shape[1] - text_size[0]) // 2
            y = y_start + i * 30
            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness)
        
        return frame
    
    def run(self):
        """Run the dashboard application"""
        system_monitor.start_monitoring()
        logger.info("Starting Vehicle Monitoring Web Dashboard...")
        logger.info(f"Access the dashboard at: http://{self.host}:{self.port}")
        
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug, allow_unsafe_werkzeug=True)
