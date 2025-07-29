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
from sort import Sort

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVehicleCounter:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5, db_path='vehicle_counts.db'):
        """
        Initialize Advanced Vehicle Counter with world-class tracking and recognition
        
        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold for detection
            db_path: Path to SQLite database
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.db_path = db_path
        self.setup_database()
        
        # Enhanced vehicle classes with better classification
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            1: 'bicycle'  # Adding bicycle support
        }
        
        # Sub-classification for better vehicle type detection
        self.vehicle_subtypes = {
            'car': ['sedan', 'hatchback', 'suv', 'jeep', 'coupe'],
            'truck': ['pickup', 'delivery', 'semi', 'heavy'],
            'bus': ['city', 'school', 'coach'],
            'motorcycle': ['bike', 'scooter', 'sport']
        }
        
        # SORT tracker for advanced multi-object tracking
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Tracking variables
        self.counted_ids = set()
        self.vehicle_counts = defaultdict(int)
        self.total_count = 0
        self.license_plates = {}
        
        # Initialize EasyOCR reader for license plate recognition
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            logger.info("EasyOCR initialized with GPU support")
        except:
            self.reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR initialized with CPU support")
        
        # Speed calculation
        self.vehicle_speeds = {}
        self.position_history = defaultdict(lambda: deque(maxlen=10))
        
        # Counting line setup
        self.counting_line = None
        self.line_position = 0.6  # 60% from top
        self.counting_direction = "down"  # "up" or "down"
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        
    def setup_database(self):
        """Setup enhanced SQLite database for storing vehicle data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced vehicle_counts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                vehicle_id TEXT UNIQUE,
                vehicle_type TEXT,
                vehicle_subtype TEXT,
                license_plate TEXT,
                speed REAL,
                location TEXT,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER
            )
        ''')
        
        # Hourly summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour DATETIME,
                total_vehicles INTEGER,
                cars INTEGER,
                motorcycles INTEGER,
                buses INTEGER,
                trucks INTEGER,
                bicycles INTEGER,
                avg_speed REAL,
                unique_license_plates INTEGER
            )
        ''')
        
        # License plate tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                vehicle_type TEXT,
                first_seen DATETIME,
                last_seen DATETIME,
                count INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def classify_vehicle_subtype(self, bbox, vehicle_type):
        """Enhanced vehicle classification based on size and aspect ratio"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        area = width * height
        
        if vehicle_type == 'car':
            if aspect_ratio > 2.5:
                return 'sedan'
            elif aspect_ratio > 2.0:
                return 'suv'
            elif area > 15000:
                return 'jeep'
            else:
                return 'hatchback'
        elif vehicle_type == 'truck':
            if area > 25000:
                return 'semi'
            elif aspect_ratio > 2.0:
                return 'pickup'
            else:
                return 'delivery'
        elif vehicle_type == 'bus':
            if area > 30000:
                return 'coach'
            else:
                return 'city'
        elif vehicle_type == 'motorcycle':
            if area < 3000:
                return 'scooter'
            else:
                return 'bike'
        
        return vehicle_type
    
    def recognize_license_plate(self, frame, bbox):
        """Advanced license plate recognition with preprocessing"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Expand bbox slightly for better plate capture
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        # Extract and preprocess the region
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        # Preprocessing for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        try:
            # OCR with multiple attempts
            results = self.reader.readtext(thresh, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            if not results:
                # Try with original image if preprocessed fails
                results = self.reader.readtext(roi, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            # Extract and validate license plate
            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    # Clean and validate plate number
                    plate_number = re.sub(r'[^A-Z0-9]', '', text.upper())
                    
                    # Basic validation for license plate format
                    if len(plate_number) >= 4 and len(plate_number) <= 10:
                        return plate_number
                        
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            
        return None
    
    def calculate_speed(self, track_id, current_pos, fps=30):
        """Enhanced speed calculation with smoothing"""
        if track_id not in self.position_history:
            self.position_history[track_id].append((current_pos, time.time()))
            return 0
            
        positions = self.position_history[track_id]
        positions.append((current_pos, time.time()))
        
        if len(positions) < 3:  # Need at least 3 points for better accuracy
            return 0
            
        # Use multiple points for better speed estimation
        speeds = []
        for i in range(len(positions) - 1):
            pos1, time1 = positions[i]
            pos2, time2 = positions[i + 1]
            
            pixel_distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            time_diff = time2 - time1
            
            if time_diff > 0:
                # Calibrated conversion factor (should be adjusted based on camera setup)
                meters_per_pixel = 0.05  # This needs real-world calibration
                distance_meters = pixel_distance * meters_per_pixel
                speed_ms = distance_meters / time_diff
                speed_kmh = speed_ms * 3.6
                speeds.append(speed_kmh)
        
        # Return smoothed average speed
        if speeds:
            avg_speed = np.mean(speeds)
            return min(avg_speed, 200)  # Cap at reasonable speed
        
        return 0
    
    def save_to_database(self, vehicle_id, vehicle_type, vehicle_subtype, license_plate, 
                        speed, bbox, confidence):
        """Save comprehensive vehicle data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert vehicle count
            cursor.execute('''
                INSERT OR REPLACE INTO vehicle_counts 
                (timestamp, vehicle_id, vehicle_type, vehicle_subtype, license_plate, 
                 speed, location, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), str(vehicle_id), vehicle_type, vehicle_subtype, 
                  license_plate, speed, 'Main Road', confidence, 
                  int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            
            # Update license plate tracking if plate detected
            if license_plate:
                cursor.execute('''
                    INSERT OR REPLACE INTO license_plates 
                    (plate_number, vehicle_type, first_seen, last_seen, count)
                    VALUES (?, ?, 
                           COALESCE((SELECT first_seen FROM license_plates WHERE plate_number = ?), ?),
                           ?, 
                           COALESCE((SELECT count FROM license_plates WHERE plate_number = ?), 0) + 1)
                ''', (license_plate, vehicle_type, license_plate, datetime.now(), 
                      datetime.now(), license_plate))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Database error: {e}")
        finally:
            conn.close()
    
    def process_detections(self, detections, frame):
        """Process YOLO detections with advanced tracking"""
        if detections is None or len(detections) == 0:
            # Update tracker with empty detections
            tracked_objects = self.tracker.update(np.empty((0, 5)))
            return []
        
        # Convert detections to format expected by SORT tracker
        detection_list = []
        
        for detection in detections:
            confidence_val = float(detection.conf.cpu().numpy())
            if confidence_val < self.confidence:
                continue
                
            cls = int(detection.cls.cpu().numpy())
            if cls not in self.vehicle_classes:
                continue
                
            bbox_coords = detection.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = bbox_coords
            confidence = confidence_val
            
            # Format: [x1, y1, x2, y2, confidence]
            detection_list.append([x1, y1, x2, y2, confidence])
        
        # Update tracker
        if detection_list:
            tracked_objects = self.tracker.update(np.array(detection_list))
        else:
            tracked_objects = self.tracker.update(np.empty((0, 5)))
        
        # Process tracked objects
        tracked_vehicles = []
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            bbox = [x1, y1, x2, y2]
            
            # Find corresponding detection for classification
            vehicle_type = 'unknown'
            confidence = 0.0
            
            for i, detection in enumerate(detections):
                confidence_val = float(detection.conf.cpu().numpy())
                if confidence_val < self.confidence:
                    continue
                    
                det_bbox = detection.xyxy[0].cpu().numpy()
                
                # Calculate IoU to match detection with track
                if self.calculate_iou(bbox, det_bbox) > 0.5:
                    cls = int(detection.cls.cpu().numpy())
                    if cls in self.vehicle_classes:
                        vehicle_type = self.vehicle_classes[cls]
                        confidence = confidence_val
                    break
            
            if vehicle_type != 'unknown':
                tracked_vehicles.append({
                    'id': int(track_id),
                    'bbox': bbox,
                    'type': vehicle_type,
                    'confidence': confidence
                })
        
        return tracked_vehicles
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def check_line_crossing(self, tracked_vehicles, frame_shape):
        """Check if vehicles crossed the counting line with direction awareness"""
        if self.counting_line is None:
            height = frame_shape[0]
            self.counting_line = int(height * self.line_position)
        
        newly_counted = []
        
        for vehicle in tracked_vehicles:
            track_id = vehicle['id']
            bbox = vehicle['bbox']
            vehicle_type = vehicle['type']
            confidence = vehicle['confidence']
            
            # Get vehicle center
            center_y = int((bbox[1] + bbox[3]) / 2)
            center_x = int((bbox[0] + bbox[2]) / 2)
            
            # Check if vehicle crossed the line and hasn't been counted
            if track_id not in self.counted_ids:
                line_crossed = False
                
                if self.counting_direction == "down" and center_y > self.counting_line:
                    line_crossed = True
                elif self.counting_direction == "up" and center_y < self.counting_line:
                    line_crossed = True
                
                if line_crossed:
                    # Vehicle crossed the line - count it
                    self.vehicle_counts[vehicle_type] += 1
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                    
                    # Enhanced classification
                    vehicle_subtype = self.classify_vehicle_subtype(bbox, vehicle_type)
                    
                    # License plate recognition
                    license_plate = self.recognize_license_plate(frame_shape, bbox)
                    if license_plate:
                        self.license_plates[track_id] = license_plate
                    
                    # Calculate speed
                    speed = self.calculate_speed(track_id, (center_x, center_y))
                    self.vehicle_speeds[track_id] = speed
                    
                    # Save to database
                    self.save_to_database(track_id, vehicle_type, vehicle_subtype, 
                                        license_plate, speed, bbox, confidence)
                    
                    newly_counted.append({
                        'id': track_id,
                        'type': vehicle_type,
                        'subtype': vehicle_subtype,
                        'license_plate': license_plate,
                        'speed': speed
                    })
                    
                    logger.info(f"Vehicle {track_id} ({vehicle_type}/{vehicle_subtype}) counted. "
                              f"Speed: {speed:.1f} km/h, Plate: {license_plate or 'N/A'}")
        
        return newly_counted
    
    def draw_enhanced_annotations(self, frame, tracked_vehicles):
        """Draw enhanced annotations with comprehensive information"""
        height, width = frame.shape[:2]
        
        # Draw counting line
        if self.counting_line is None:
            self.counting_line = int(height * self.line_position)
        
        cv2.line(frame, (0, self.counting_line), (width, self.counting_line), (0, 0, 255), 3)
        cv2.putText(frame, f'COUNTING LINE ({self.counting_direction.upper()})', 
                   (10, self.counting_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw vehicle detections
        for vehicle in tracked_vehicles:
            track_id = vehicle['id']
            bbox = vehicle['bbox']
            vehicle_type = vehicle['type']
            confidence = vehicle['confidence']
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Color coding: Green for counted, Blue for tracking
            color = (0, 255, 0) if track_id in self.counted_ids else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label with comprehensive info
            speed = self.vehicle_speeds.get(track_id, 0)
            license_plate = self.license_plates.get(track_id, '')
            
            label_lines = [
                f"ID:{track_id} {vehicle_type.upper()}",
                f"Conf:{confidence:.2f}"
            ]
            
            if speed > 0:
                label_lines.append(f"Speed:{speed:.1f}km/h")
            
            if license_plate:
                label_lines.append(f"Plate:{license_plate}")
            
            # Draw multi-line label
            label_y = y1 - 10
            for line in label_lines:
                cv2.putText(frame, line, (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                label_y -= 15
        
        # Draw comprehensive statistics
        self.draw_enhanced_statistics(frame)
        
        return frame
    
    def draw_enhanced_statistics(self, frame):
        """Draw comprehensive statistics panel"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for statistics
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 350, 10), (width - 10, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        x_offset = width - 340
        
        # Title
        cv2.putText(frame, 'ADVANCED VEHICLE MONITORING', (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Total count
        cv2.putText(frame, f'Total Vehicles: {self.total_count}', (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # Individual counts
        for vehicle_type, count in self.vehicle_counts.items():
            cv2.putText(frame, f'{vehicle_type.title()}: {count}', (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 18
        
        # License plates detected
        plate_count = len(self.license_plates)
        cv2.putText(frame, f'License Plates: {plate_count}', (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y_offset += 18
        
        # Performance metrics
        if self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f'FPS: {fps:.1f}', (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f'Time: {current_time}', (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def process_video(self, source=0, output_path=None, display=True):
        """
        Process video source with advanced vehicle counting
        
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
        logger.info("Advanced Vehicle Counter with:")
        logger.info("✓ Multi-Object Tracking (SORT)")
        logger.info("✓ License Plate Recognition (EasyOCR)")
        logger.info("✓ Enhanced Vehicle Classification")
        logger.info("✓ Speed Estimation")
        logger.info("✓ Duplicate Prevention")
        
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                detections = results[0].boxes
                
                # Process detections with advanced tracking
                tracked_vehicles = self.process_detections(detections, frame)
                
                # Check line crossing and count vehicles
                newly_counted = self.check_line_crossing(tracked_vehicles, frame)
                
                # Draw enhanced annotations
                frame = self.draw_enhanced_annotations(frame, tracked_vehicles)
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Advanced Vehicle Counter', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):  # Reset counts
                        self.reset_counts()
                        logger.info("Counts reset")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Print comprehensive final statistics
            self.print_final_statistics()
    
    def reset_counts(self):
        """Reset all counting variables"""
        self.counted_ids.clear()
        self.vehicle_counts.clear()
        self.total_count = 0
        self.license_plates.clear()
        self.vehicle_speeds.clear()
        self.position_history.clear()
    
    def print_final_statistics(self):
        """Print comprehensive final statistics"""
        logger.info("=" * 50)
        logger.info("FINAL COMPREHENSIVE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total vehicles counted: {self.total_count}")
        
        for vehicle_type, count in self.vehicle_counts.items():
            logger.info(f"{vehicle_type.title()}: {count}")
        
        logger.info(f"Unique license plates detected: {len(self.license_plates)}")
        
        if self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"Average FPS: {avg_fps:.2f}")
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        
        # Database statistics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vehicle_counts")
        db_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT license_plate) FROM vehicle_counts WHERE license_plate IS NOT NULL")
        db_plates = cursor.fetchone()[0]
        conn.close()
        
        logger.info(f"Records in database: {db_count}")
        logger.info(f"Unique plates in database: {db_plates}")
        logger.info("=" * 50)
    
    def get_advanced_summary(self, hours=24):
        """Generate advanced summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get data for specified time period
        start_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT 
                vehicle_type,
                vehicle_subtype,
                COUNT(*) as count,
                AVG(speed) as avg_speed,
                COUNT(DISTINCT license_plate) as unique_plates
            FROM vehicle_counts 
            WHERE timestamp >= ?
            GROUP BY vehicle_type, vehicle_subtype
            ORDER BY count DESC
        ''', (start_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        summary = {
            'period_hours': hours,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'vehicle_details': [],
            'total_vehicles': sum(row[2] for row in results),
            'total_unique_plates': sum(row[4] for row in results)
        }
        
        for row in results:
            vehicle_type, vehicle_subtype, count, avg_speed, unique_plates = row
            summary['vehicle_details'].append({
                'type': vehicle_type,
                'subtype': vehicle_subtype,
                'count': count,
                'avg_speed': avg_speed or 0,
                'unique_plates': unique_plates
            })
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Advanced AI-Powered Vehicle Monitoring System')
    parser.add_argument('--source', default=0, help='Video source (camera index, file path, or RTSP URL)')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--db-path', default='advanced_vehicle_counts.db', help='Database path')
    parser.add_argument('--line-position', type=float, default=0.6, help='Counting line position (0.0-1.0)')
    parser.add_argument('--direction', choices=['up', 'down'], default='down', help='Counting direction')
    
    args = parser.parse_args()
    
    # Initialize advanced vehicle counter
    counter = AdvancedVehicleCounter(
        model_path=args.model,
        confidence=args.confidence,
        db_path=args.db_path
    )
    
    # Set counting parameters
    counter.line_position = args.line_position
    counter.counting_direction = args.direction
    
    # Process video
    counter.process_video(
        source=args.source,
        output_path=args.output,
        display=not args.no_display
    )

if __name__ == "__main__":
    main()
