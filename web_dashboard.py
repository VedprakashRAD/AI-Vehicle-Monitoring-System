from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import cv2
import sqlite3
import json
import threading
import time
from datetime import datetime, timedelta
import base64
import numpy as np
from vehicle_counter import VehicleCounter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vehicle_monitoring_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
vehicle_counter = None
current_frame = None
processing_thread = None
is_processing = False

class WebVehicleCounter(VehicleCounter):
    """Extended VehicleCounter for web dashboard"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_frame = None
        self.latest_stats = {}
        
    def process_frame_for_web(self, frame):
        """Process single frame and return annotated frame with stats"""
        global current_frame
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        detections = results[0].boxes
        
        if detections is not None:
            # Track vehicles
            self.track_vehicles(detections, frame.shape)
            
            # Check line crossing
            self.check_line_crossing(frame.shape)
            
            # Draw annotations
            frame = self.draw_annotations(frame, detections)
        
        # Update latest stats
        self.latest_stats = {
            'total_count': self.total_count,
            'vehicle_counts': dict(self.vehicle_counts),
            'timestamp': datetime.now().isoformat(),
            'active_tracks': len(self.tracks)
        }
        
        # Store current frame
        current_frame = frame
        return frame, self.latest_stats

def generate_frames():
    """Generate video frames for streaming"""
    global vehicle_counter, current_frame
    
    if vehicle_counter is None:
        return
        
    cap = cv2.VideoCapture(0)  # Default camera
    
    try:
        while True:
            if not is_processing:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, stats = vehicle_counter.process_frame_for_web(frame)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Emit stats via WebSocket
            socketio.emit('stats_update', stats)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        logger.error(f"Error in frame generation: {e}")
    finally:
        cap.release()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start vehicle monitoring"""
    global vehicle_counter, processing_thread, is_processing
    
    try:
        data = request.get_json()
        source = data.get('source', 0)
        confidence = float(data.get('confidence', 0.5))
        
        # Initialize vehicle counter
        vehicle_counter = WebVehicleCounter(confidence=confidence)
        is_processing = True
        
        logger.info(f"Started monitoring with source: {source}, confidence: {confidence}")
        return jsonify({'status': 'success', 'message': 'Monitoring started'})
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop vehicle monitoring"""
    global is_processing
    
    is_processing = False
    logger.info("Monitoring stopped")
    return jsonify({'status': 'success', 'message': 'Monitoring stopped'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    if vehicle_counter:
        return jsonify(vehicle_counter.latest_stats)
    return jsonify({'error': 'No active monitoring'})

@app.route('/api/history')
def get_history():
    """Get historical data"""
    try:
        hours = int(request.args.get('hours', 24))
        conn = sqlite3.connect('vehicle_counts.db')
        cursor = conn.cursor()
        
        # Get data for specified hours
        start_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT timestamp, vehicle_type, speed
            FROM vehicle_counts 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (start_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Format data for charts
        history_data = []
        for timestamp, vehicle_type, speed in results:
            history_data.append({
                'timestamp': timestamp,
                'vehicle_type': vehicle_type,
                'speed': speed
            })
        
        return jsonify(history_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/hourly_summary')
def get_hourly_summary():
    """Get hourly summary data"""
    try:
        days = int(request.args.get('days', 7))
        conn = sqlite3.connect('vehicle_counts.db')
        cursor = conn.cursor()
        
        start_time = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                vehicle_type,
                COUNT(*) as count,
                AVG(speed) as avg_speed
            FROM vehicle_counts 
            WHERE timestamp >= ?
            GROUP BY hour, vehicle_type
            ORDER BY hour DESC
        ''', (start_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Group by hour
        hourly_data = {}
        for hour, vehicle_type, count, avg_speed in results:
            if hour not in hourly_data:
                hourly_data[hour] = {
                    'hour': hour,
                    'total': 0,
                    'vehicles': {},
                    'avg_speed': 0
                }
            
            hourly_data[hour]['vehicles'][vehicle_type] = count
            hourly_data[hour]['total'] += count
            hourly_data[hour]['avg_speed'] = avg_speed or 0
        
        return jsonify(list(hourly_data.values()))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/export_data')
def export_data():
    """Export data as CSV"""
    try:
        conn = sqlite3.connect('vehicle_counts.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, vehicle_type, speed, location
            FROM vehicle_counts 
            ORDER BY timestamp DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        # Create CSV content
        csv_content = "Timestamp,Vehicle Type,Speed (km/h),Location\n"
        for timestamp, vehicle_type, speed, location in results:
            csv_content += f"{timestamp},{vehicle_type},{speed or 0},{location}\n"
        
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=vehicle_data.csv"}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to vehicle monitoring system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_stats')
def handle_stats_request():
    """Handle stats request from client"""
    if vehicle_counter:
        emit('stats_update', vehicle_counter.latest_stats)

def create_templates():
    """Create HTML templates"""
    import os
    
    # Create templates directory
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    # Create index.html
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Vehicle Monitoring System</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            grid-template-rows: auto auto;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .video-panel {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            grid-row: span 2;
        }
        
        .stats-panel {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .controls-panel {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .video-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            aspect-ratio: 16/9;
        }
        
        #videoFeed {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .stat-value {
            font-weight: bold;
            color: #667eea;
        }
        
        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        
        .control-group label {
            font-weight: bold;
            color: #555;
        }
        
        .control-group input, .control-group select {
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
        }
        
        .charts-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .chart-panel {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        .status-offline {
            background: #f44336;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .export-section {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            margin-top: 20px;
            text-align: center;
        }
        
        @media (max-width: 1200px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            .charts-section {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 AI Vehicle Monitoring System</h1>
            <p>Real-time CCTV-based Vehicle Detection and Counting</p>
        </div>
        
        <div class="dashboard-grid">
            <div class="video-panel">
                <h3>Live Video Feed</h3>
                <div class="video-container">
                    <img id="videoFeed" src="/video_feed" alt="Video Feed" style="display: none;">
                    <div id="videoPlaceholder" style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;">
                        Click "Start Monitoring" to begin video feed
                    </div>
                </div>
            </div>
            
            <div class="stats-panel">
                <h3><span id="statusIndicator" class="status-indicator status-offline"></span>Live Statistics</h3>
                <div id="statsContainer">
                    <div class="stat-item">
                        <span>Total Vehicles:</span>
                        <span class="stat-value" id="totalCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Cars:</span>
                        <span class="stat-value" id="carCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Motorcycles:</span>
                        <span class="stat-value" id="motorcycleCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Buses:</span>
                        <span class="stat-value" id="busCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Trucks:</span>
                        <span class="stat-value" id="truckCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Active Tracks:</span>
                        <span class="stat-value" id="activeCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span>Last Updated:</span>
                        <span class="stat-value" id="lastUpdated">Never</span>
                    </div>
                </div>
            </div>
            
            <div class="controls-panel">
                <h3>Control Panel</h3>
                <div class="controls">
                    <div class="control-group">
                        <label>Video Source:</label>
                        <select id="videoSource">
                            <option value="0">Default Camera</option>
                            <option value="1">Camera 1</option>
                            <option value="test_video.mp4">Test Video</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label>Confidence Threshold:</label>
                        <input type="range" id="confidence" min="0.1" max="1.0" step="0.1" value="0.5">
                        <span id="confidenceValue">0.5</span>
                    </div>
                    
                    <button class="btn btn-primary" onclick="startMonitoring()">Start Monitoring</button>
                    <button class="btn btn-danger" onclick="stopMonitoring()">Stop Monitoring</button>
                </div>
            </div>
        </div>
        
        <div class="charts-section">
            <div class="chart-panel">
                <h3>Vehicle Count Over Time</h3>
                <canvas id="timeChart"></canvas>
            </div>
            
            <div class="chart-panel">
                <h3>Vehicle Type Distribution</h3>
                <canvas id="typeChart"></canvas>
            </div>
        </div>
        
        <div class="export-section">
            <h3>Data Export</h3>
            <p>Download historical vehicle data for analysis</p>
            <button class="btn btn-primary" onclick="exportData()">Export CSV Data</button>
        </div>
    </div>

    <script>
        const socket = io();
        let timeChart, typeChart;
        let isMonitoring = false;
        
        // Initialize charts
        function initCharts() {
            const timeCtx = document.getElementById('timeChart').getContext('2d');
            timeChart = new Chart(timeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Vehicles per Hour',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            const typeCtx = document.getElementById('typeChart').getContext('2d');
            typeChart = new Chart(typeCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Cars', 'Motorcycles', 'Buses', 'Trucks'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                    }]
                },
                options: {
                    responsive: true
                }
            });
        }
        
        // Update confidence value display
        document.getElementById('confidence').addEventListener('input', function() {
            document.getElementById('confidenceValue').textContent = this.value;
        });
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('stats_update', function(data) {
            updateStats(data);
        });
        
        function updateStats(data) {
            document.getElementById('totalCount').textContent = data.total_count || 0;
            document.getElementById('carCount').textContent = data.vehicle_counts.car || 0;
            document.getElementById('motorcycleCount').textContent = data.vehicle_counts.motorcycle || 0;
            document.getElementById('busCount').textContent = data.vehicle_counts.bus || 0;
            document.getElementById('truckCount').textContent = data.vehicle_counts.truck || 0;
            document.getElementById('activeCount').textContent = data.active_tracks || 0;
            
            const lastUpdated = new Date(data.timestamp).toLocaleTimeString();
            document.getElementById('lastUpdated').textContent = lastUpdated;
            
            // Update type chart
            if (typeChart) {
                typeChart.data.datasets[0].data = [
                    data.vehicle_counts.car || 0,
                    data.vehicle_counts.motorcycle || 0,
                    data.vehicle_counts.bus || 0,
                    data.vehicle_counts.truck || 0
                ];
                typeChart.update();
            }
        }
        
        function startMonitoring() {
            const source = document.getElementById('videoSource').value;
            const confidence = document.getElementById('confidence').value;
            
            fetch('/start_monitoring', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    source: source,
                    confidence: confidence
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    isMonitoring = true;
                    document.getElementById('statusIndicator').className = 'status-indicator status-online';
                    document.getElementById('videoFeed').style.display = 'block';
                    document.getElementById('videoPlaceholder').style.display = 'none';
                    alert('Monitoring started successfully!');
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Failed to start monitoring');
            });
        }
        
        function stopMonitoring() {
            fetch('/stop_monitoring', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    isMonitoring = false;
                    document.getElementById('statusIndicator').className = 'status-indicator status-offline';
                    document.getElementById('videoFeed').style.display = 'none';
                    document.getElementById('videoPlaceholder').style.display = 'flex';
                    alert('Monitoring stopped');
                }
            });
        }
        
        function exportData() {
            window.open('/api/export_data', '_blank');
        }
        
        // Load historical data for charts
        function loadHistoricalData() {
            fetch('/api/hourly_summary?days=1')
            .then(response => response.json())
            .then(data => {
                if (timeChart && data.length > 0) {
                    const labels = data.map(item => new Date(item.hour).toLocaleTimeString());
                    const counts = data.map(item => item.total);
                    
                    timeChart.data.labels = labels.reverse();
                    timeChart.data.datasets[0].data = counts.reverse();
                    timeChart.update();
                }
            });
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            loadHistoricalData();
            
            // Refresh historical data every 5 minutes
            setInterval(loadHistoricalData, 300000);
        });
    </script>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'index.html'), 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    # Create templates
    create_templates()
    
    # Run the web application
    logger.info("Starting Vehicle Monitoring Web Dashboard...")
    logger.info("Access the dashboard at: http://localhost:8080")
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
