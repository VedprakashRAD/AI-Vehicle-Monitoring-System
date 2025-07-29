import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime, timedelta
import json
import threading
import time
from collections import defaultdict, deque
import argparse
import logging
import easyocr
import re
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import uuid
import pickle
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleCounter:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5, db_path='vehicle_counts.db'):
        """
        Initialize Vehicle Counter
        
        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold for detection
            db_path: Path to SQLite database
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.db_path = db_path
        self.setup_database()
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Tracking variables
self.kalman_filters = {}
self.tracks = {}
self.next_id = uuid.uuid4  # Use UUID for unique IDs
self.counted_ids = set()
self.vehicle_counts = defaultdict(int)
self.total_count = 0

# EasyOCR reader
self.reader = easyocr.Reader(['en'])  # Use with English character detection

# For speed calculation
self.vehicle_speeds = {}
self.position_history = defaultdict(lambda: deque(maxlen=10))

# Initialize license plate history to avoid duplicates
self.license_plate_history = {}
        
        # Line coordinates for counting (will be set dynamically)
        self.counting_line = None
        self.line_position = 0.6  # 60% from top
        
    def setup_database(self):
        """Setup SQLite database for storing vehicle counts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                vehicle_type TEXT,
                count INTEGER,
                location TEXT,
                speed REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour DATETIME,
                total_vehicles INTEGER,
                cars INTEGER,
                motorcycles INTEGER,
                buses INTEGER,
                trucks INTEGER,
                avg_speed REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_to_database(self, vehicle_type, speed=None):
        """Save vehicle count to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO vehicle_counts (timestamp, vehicle_type, count, location, speed)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), vehicle_type, 1, 'Main Road', speed))
        
        conn.commit()
        conn.close()
        
def allocate_kalman_filter(self, object_id, initial_pos):
    """ Allocate a new Kalman filter for object tracking """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([initial_pos[0], initial_pos[1], 0, 0])  # Initial state (location and velocity)
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.R *= 10  # Measurement uncertainty
    kf.P *= 100  # Initial uncertainty
    kf.Q *= 0.01  # Process noise
    self.kalman_filters[object_id] = kf





def calculate_distance(self, p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_speed(self, vehicle_id, current_pos, fps=30):
        """Calculate vehicle speed in km/h"""
        if vehicle_id not in self.position_history:
            self.position_history[vehicle_id].append((current_pos, time.time()))
            return 0
            
        positions = self.position_history[vehicle_id]
        positions.append((current_pos, time.time()))
        
        if len(positions) < 2:
            return 0
            
        # Calculate speed using last two positions
        pos1, time1 = positions[-2]
        pos2, time2 = positions[-1]
        
        pixel_distance = self.calculate_distance(pos1, pos2)
        time_diff = time2 - time1
        
        if time_diff == 0:
            return 0
            
        # Convert pixel distance to meters (approximate)
        # This should be calibrated based on camera setup
        meters_per_pixel = 0.05  # Approximate conversion
        distance_meters = pixel_distance * meters_per_pixel
        
        # Calculate speed in m/s then convert to km/h
        speed_ms = distance_meters / time_diff
        speed_kmh = speed_ms * 3.6
        
        return min(speed_kmh, 200)  # Cap at reasonable speed
    
    def track_vehicles(self, detections, frame_shape):
        """Simple tracking algorithm using center point distance"""
        current_tracks = {}
        
        for detection in detections:
            if detection.conf < self.confidence:
                continue
                
            cls = int(detection.cls)
            if cls not in self.vehicle_classes:
                continue
                
            # Get bounding box center
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Find closest existing track
            min_distance = float('inf')
            closest_id = None
            
for track_id, (prev_x, prev_y, prev_cls) in self.tracks.items():
    if prev_cls == cls:  # Same vehicle type
        # Use Kalman filter for tracking
        kf = self.kalman_filters.get(track_id)
        if kf is not None:
            kf.predict()
            distance = np.linalg.norm(kf.x[:2] - np.array([center_x, center_y]))
            if distance < min_distance and distance < 100:  # Improved max tracking distance
                min_distance = distance
                closest_id = track_id
if closest_id is not None:
    # Update existing track
    current_tracks[closest_id] = (center_x, center_y, cls)

    # Update Kalman filter with new position
    kf = self.kalman_filters[closest_id]
    kf.update(np.array([center_x, center_y]))

    # Calculate speed
    speed = self.calculate_speed(closest_id, (center_x, center_y))
    self.vehicle_speeds[closest_id] = speed
else:
    # Create new track with Kalman filter
    new_id = self.next_id()
    self.kalman_filters[new_id] = self.allocate_kalman_filter(new_id, np.array([center_x, center_y]))
    current_tracks[new_id] = (center_x, center_y, cls)
                self.next_id += 1
        
        self.tracks = current_tracks
        return current_tracks
    
    def check_line_crossing(self, frame_shape):
        """Check if vehicles crossed the counting line"""
        if self.counting_line is None:
            height = frame_shape[0]
            self.counting_line = int(height * self.line_position)
        
        for track_id, (x, y, cls) in self.tracks.items():
            if track_id not in self.counted_ids:
                if y > self.counting_line:  # Vehicle crossed the line
                    vehicle_type = self.class_names[cls]
                    self.vehicle_counts[vehicle_type] += 1
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    
                    # Save to database
                    speed = self.vehicle_speeds.get(track_id, 0)
                    self.save_to_database(vehicle_type, speed)
                    
                    logger.info(f"Vehicle {track_id} ({vehicle_type}) counted. Speed: {speed:.1f} km/h")
    
def recognize_license_plate(self, frame, bbox):
    """Recognize and return the license plate number from the bounding box"""
    x1, y1, x2, y2 = bbox
    license_plate_img = frame[y1:y2, x1:x2]
    results = self.reader.readtext(license_plate_img)

    plate_number = ""
    for (_, text, _) in results:
        plate_number += text

    # Validate plate number (simple regex for a generic plate format)
    if re.match(r'^[A-Z0-9]+$', plate_number):
        return plate_number
    return None




def draw_annotations(self, frame, detections):
    """Draw bounding boxes, tracking info, and counting line"""
    height, width = frame.shape[:2]

    # Draw counting line
    if self.counting_line is None:
        self.counting_line = int(height * self.line_position)

    cv2.line(frame, (0, self.counting_line), (width, self.counting_line), (0, 0, 255), 3)
    cv2.putText(frame, 'COUNTING LINE', (10, self.counting_line - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw detections
        for detection in detections:
            if detection.conf < self.confidence:
                continue
                
            cls = int(detection.cls)
            if cls not in self.vehicle_classes:
                continue
                
            x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
            
            # Find corresponding track
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            track_id = None
            for tid, (tx, ty, tcls) in self.tracks.items():
                if abs(tx - center_x) < 50 and abs(ty - center_y) < 50 and tcls == cls:
                    track_id = tid
                    break
            
            # Draw bounding box
            color = (0, 255, 0) if track_id in self.counted_ids else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw labels
            vehicle_type = self.class_names[cls]
            confidence = detection.conf.item()
            speed = self.vehicle_speeds.get(track_id, 0) if track_id else 0
            
            label = f"{vehicle_type} ID:{track_id} {confidence:.2f}"
            if speed > 0:
                label += f" {speed:.1f}km/h"
            
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw statistics
        self.draw_statistics(frame)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw vehicle count statistics on frame"""
        y_offset = 30
        
        # Title
        cv2.putText(frame, 'VEHICLE MONITORING SYSTEM', (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 40
        
        # Total count
        cv2.putText(frame, f'Total Vehicles: {self.total_count}', (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        # Individual counts
        for vehicle_type, count in self.vehicle_counts.items():
            cv2.putText(frame, f'{vehicle_type.title()}: {count}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        # Current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_video(self, source=0, output_path=None, display=True):
        """
        Process video source for vehicle counting
        
        Args:
            source: Video source (camera index, video file path, or RTSP URL)
            output_path: Path to save output video
            display: Whether to display video window
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Error opening video source: {source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Processing video: {width}x{height} @ {fps} FPS")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Vehicle Counter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            logger.info("=== FINAL STATISTICS ===")
            logger.info(f"Total vehicles counted: {self.total_count}")
            for vehicle_type, count in self.vehicle_counts.items():
                logger.info(f"{vehicle_type.title()}: {count}")
    
    def get_hourly_summary(self):
        """Generate hourly summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counts for current hour
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        next_hour = current_hour + timedelta(hours=1)
        
        cursor.execute('''
            SELECT vehicle_type, COUNT(*), AVG(speed)
            FROM vehicle_counts 
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY vehicle_type
        ''', (current_hour, next_hour))
        
        results = cursor.fetchall()
        conn.close()
        
        summary = {
            'hour': current_hour.isoformat(),
            'total_vehicles': sum(count for _, count, _ in results),
            'vehicles': {vehicle_type: count for vehicle_type, count, _ in results},
            'avg_speeds': {vehicle_type: speed for vehicle_type, _, speed in results if speed}
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Vehicle Monitoring System')
    parser.add_argument('--source', default=0, help='Video source (camera index, file path, or RTSP URL)')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--db-path', default='vehicle_counts.db', help='Database path')
    
    args = parser.parse_args()
    
    # Initialize vehicle counter
    counter = VehicleCounter(
        model_path=args.model,
        confidence=args.confidence,
        db_path=args.db_path
    )
    
    # Process video
    counter.process_video(
        source=args.source,
        output_path=args.output,
        display=not args.no_display
    )

if __name__ == "__main__":
    main()
