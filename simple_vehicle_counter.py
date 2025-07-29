import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVehicleCounter:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5, db_path='vehicle_counts.db'):
        """
        Initialize Simple Vehicle Counter
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.db_path = db_path
        self.setup_database()
        
        # Vehicle classes for COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Tracking variables
        self.vehicle_counts = defaultdict(int)
        self.total_count = 0
        
        # Counting line setup
        self.counting_line = None
        self.line_position = 0.6  # 60% from top
        
        # Frame tracking
        self.frame_count = 0
        
    def setup_database(self):
        """Setup simple SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                vehicle_type TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def process_frame_for_web(self, frame):
        """Process single frame and return annotated frame with stats"""
        height, width = frame.shape[:2]
        
        # Set counting line if not set
        if self.counting_line is None:
            self.counting_line = int(height * self.line_position)
        
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Process detections
            if results and len(results) > 0:
                detections = results[0].boxes
                
                if detections is not None:
                    for detection in detections:
                        # Get confidence score
                        try:
                            conf_tensor = detection.conf
                            if hasattr(conf_tensor, 'item'):
                                confidence = conf_tensor.item()
                            else:
                                confidence = float(conf_tensor)
                        except:
                            continue
                            
                        if confidence < self.confidence:
                            continue
                            
                        # Get class
                        try:
                            cls_tensor = detection.cls
                            if hasattr(cls_tensor, 'item'):
                                cls = int(cls_tensor.item())
                            else:
                                cls = int(cls_tensor)
                        except:
                            continue
                            
                        if cls not in self.vehicle_classes:
                            continue
                            
                        # Get bounding box
                        try:
                            bbox_tensor = detection.xyxy[0]
                            if hasattr(bbox_tensor, 'cpu'):
                                bbox = bbox_tensor.cpu().numpy()
                            else:
                                bbox = bbox_tensor
                            x1, y1, x2, y2 = bbox
                        except:
                            continue
                        
                        vehicle_type = self.vehicle_classes[cls]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{vehicle_type}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Simple counting based on center crossing line
                        center_y = int((y1 + y2) / 2)
                        if center_y > self.counting_line:
                            # Count vehicle (simplified - just count all detections)
                            self.vehicle_counts[vehicle_type] += 1
                            self.total_count += 1
                            
                            # Save to database
                            self.save_to_database(vehicle_type, confidence)
            
            # Draw counting line
            cv2.line(frame, (0, self.counting_line), (width, self.counting_line), (0, 0, 255), 3)
            cv2.putText(frame, 'COUNTING LINE', (10, self.counting_line - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw statistics
            self.draw_statistics(frame)
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            # Draw error message on frame
            cv2.putText(frame, f"Processing Error: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        self.frame_count += 1
        
        # Update latest stats
        latest_stats = {
            'total_count': self.total_count,
            'vehicle_counts': dict(self.vehicle_counts),
            'timestamp': datetime.now().isoformat(),
            'active_tracks': 0  # Simplified - no tracking
        }
        
        return frame, latest_stats
    
    def save_to_database(self, vehicle_type, confidence):
        """Save vehicle detection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO vehicle_counts (timestamp, vehicle_type, confidence)
                VALUES (?, ?, ?)
            ''', (datetime.now(), vehicle_type, confidence))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database error: {e}")
    
    def draw_statistics(self, frame):
        """Draw simple statistics panel"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 300, 10), (width - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw statistics
        y_offset = 40
        x_offset = width - 290
        
        cv2.putText(frame, 'VEHICLE MONITORING', (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f'Total: {self.total_count}', (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
        
        for vehicle_type, count in self.vehicle_counts.items():
            cv2.putText(frame, f'{vehicle_type.title()}: {count}', (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 20
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f'Time: {current_time}', (x_offset, y_offset + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
