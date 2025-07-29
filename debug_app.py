#!/usr/bin/env python3
"""
Debug version of vehicle monitoring web app
"""

from flask import Flask, Response, render_template_string
import cv2
import time
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable
camera_active = False

def generate_frames():
    """Generate frames for video streaming"""
    global camera_active
    
    logger.info("🎥 Starting camera capture...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("❌ Cannot open camera")
        return
    
    logger.info("✅ Camera opened successfully")
    
    try:
        while camera_active:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("⚠️ Failed to read frame")
                break
            
            # Add debug info to frame
            cv2.putText(frame, f"Frame: {int(time.time())}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Camera is working!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        logger.error(f"❌ Error in frame generation: {e}")
    finally:
        cap.release()
        logger.info("📹 Camera released")

@app.route('/')
def index():
    """Main page"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Debug Camera Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .video-container { 
                border: 2px solid #333; 
                padding: 10px; 
                margin: 20px 0;
                background: #000;
            }
            img { width: 100%; max-width: 640px; }
            button { 
                padding: 15px 30px; 
                font-size: 16px; 
                margin: 10px; 
                cursor: pointer;
            }
            .start { background: #4CAF50; color: white; border: none; }
            .stop { background: #f44336; color: white; border: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Camera Debug Test</h1>
            <p>This is a simple test to check if the camera video feed works.</p>
            
            <button class="start" onclick="startCamera()">▶️ Start Camera</button>
            <button class="stop" onclick="stopCamera()">⏹️ Stop Camera</button>
            
            <div class="video-container">
                <img id="videoFeed" src="" alt="Video will appear here" style="display: none;">
                <div id="placeholder" style="color: white; text-align: center; padding: 50px;">
                    Click "Start Camera" to begin
                </div>
            </div>
            
            <div id="status" style="margin-top: 20px; padding: 10px; background: #f0f0f0;">
                Status: Ready
            </div>
        </div>
        
        <script>
            function startCamera() {
                fetch('/start_camera', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        document.getElementById('videoFeed').src = '/video_feed';
                        document.getElementById('videoFeed').style.display = 'block';
                        document.getElementById('placeholder').style.display = 'none';
                        document.getElementById('status').innerText = 'Status: Camera Active';
                        document.getElementById('status').style.background = '#d4edda';
                    } else {
                        document.getElementById('status').innerText = 'Status: Error - ' + data.message;
                        document.getElementById('status').style.background = '#f8d7da';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('status').innerText = 'Status: Connection Error';
                    document.getElementById('status').style.background = '#f8d7da';
                });
            }
            
            function stopCamera() {
                fetch('/stop_camera', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('videoFeed').style.display = 'none';
                    document.getElementById('placeholder').style.display = 'block';
                    document.getElementById('status').innerText = 'Status: Camera Stopped';
                    document.getElementById('status').style.background = '#f0f0f0';
                });
            }
        </script>
    </body>
    </html>
    '''
    return html

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera"""
    global camera_active
    
    try:
        camera_active = True
        logger.info("🎬 Camera started")
        return {'status': 'success', 'message': 'Camera started'}
    except Exception as e:
        logger.error(f"❌ Error starting camera: {e}")
        return {'status': 'error', 'message': str(e)}

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera"""
    global camera_active
    
    camera_active = False
    logger.info("⏹️ Camera stopped")
    return {'status': 'success', 'message': 'Camera stopped'}

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    logger.info("📡 Video feed requested")
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🔍 Starting Camera Debug Test...")
    print("📍 Open your browser and go to: http://localhost:8080")
    print("⏹️  Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
