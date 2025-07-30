import cv2
import numpy as np
from datetime import datetime
import sqlite3
import random
import string
import os

class ImprovedVehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.entry_exit_log = []
        self.frame_count = 0
        self.vehicle_tracker = {}
        self.next_vehicle_id = 1
        
        # Initialize detection methods
        self.init_yolo()
        self.init_background_subtractor()
        self.init_database()
        
    def init_yolo(self):
        """Initialize YOLOv8 with better configuration"""
        try:
            from ultralytics import YOLO
            # Try different YOLO models for better accuracy
            model_paths = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
            
            for model_path in model_paths:
                try:
                    self.yolo_model = YOLO(model_path)
                    self.use_yolo = True
                    print(f"✅ Loaded {model_path} successfully")
                    break
                except:
                    continue
            else:
                raise Exception("No YOLO model could be loaded")
                
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            self.yolo_model = None
            self.use_yolo = False
    
    def init_background_subtractor(self):
        """Initialize background subtraction for motion detection"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
        
    def init_database(self):
        conn = sqlite3.connect('vehicle_tracking.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT,
                registration_number TEXT,
                vehicle_type TEXT,
                status TEXT,
                entry_time DATETIME,
                exit_time DATETIME,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def generate_license_plate(self):
        """Generate realistic license plates"""
        formats = [
            lambda: f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}{''.join(random.choices('0123456789', k=3))}",
            lambda: f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{''.join(random.choices('0123456789', k=4))}",
            lambda: f"{''.join(random.choices('0123456789', k=3))}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}"
        ]
        return random.choice(formats)()
    
    def detect_with_yolo(self, frame):
        """Enhanced YOLO detection with better filtering"""
        detections = []
        if not self.use_yolo:
            return detections
            
        try:
            # Run YOLO with optimized parameters
            results = self.yolo_model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.5,
                max_det=50,
                classes=[2, 3, 5, 7]  # Only vehicle classes
            )
            
            vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in vehicle_classes and conf > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Filter out very small detections
                            width = x2 - x1
                            height = y2 - y1
                            if width > 30 and height > 30:
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'type': vehicle_classes[cls],
                                    'confidence': conf,
                                    'area': width * height
                                })
                                
        except Exception as e:
            print(f"YOLO detection error: {e}")
            
        return detections
    
    def detect_with_background_subtraction(self, frame):
        """Motion-based vehicle detection as backup"""
        detections = []
        
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter based on aspect ratio (vehicles are typically wider than tall)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 4.0:
                        detections.append({
                            'bbox': (x, y, x + w, y + h),
                            'type': 'vehicle',  # Generic type for motion detection
                            'confidence': 0.7,
                            'area': area
                        })
                        
        except Exception as e:
            print(f"Background subtraction error: {e}")
            
        return detections
    
    def track_vehicles(self, detections, frame_shape):
        """Simple vehicle tracking"""
        frame_height, frame_width = frame_shape[:2]
        entry_y = frame_height // 3
        exit_y = (frame_height * 2) // 3
        
        tracked_vehicles = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Determine status based on position
            if center_y < entry_y + 20:
                status = 'Entry'
            elif center_y > exit_y - 20:
                status = 'Exit'
            else:
                status = 'Monitoring'
            
            # Generate vehicle ID
            vehicle_id = f"VH{self.next_vehicle_id:04d}"
            self.next_vehicle_id += 1
            
            tracked_vehicles.append({
                'vehicle_id': vehicle_id,
                'bbox': detection['bbox'],
                'type': detection['type'],
                'confidence': detection['confidence'],
                'status': status,
                'center': (center_x, center_y)
            })
            
        return tracked_vehicles
    
    def process_frame_for_web(self, frame):
        self.frame_count += 1
        processed_frame = frame.copy()
        vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0, 'vehicle': 0}
        
        # Draw monitoring lines and zones
        frame_height, frame_width = processed_frame.shape[:2]
        entry_y = frame_height // 3
        exit_y = (frame_height * 2) // 3
        
        # Entry line (Green)
        cv2.line(processed_frame, (0, entry_y), (frame_width, entry_y), (0, 255, 0), 3)
        cv2.putText(processed_frame, "ENTRY LINE", (10, entry_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Exit line (Red)
        cv2.line(processed_frame, (0, exit_y), (frame_width, exit_y), (0, 0, 255), 3)
        cv2.putText(processed_frame, "EXIT LINE", (10, exit_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Monitoring zone
        cv2.rectangle(processed_frame, (10, entry_y + 10), (frame_width - 10, exit_y - 10), 
                     (255, 255, 0), 2)
        
        # Vehicle detection
        detections = []
        
        if self.use_yolo:
            yolo_detections = self.detect_with_yolo(frame)
            detections.extend(yolo_detections)
            cv2.putText(processed_frame, "YOLOv8 AI Detection Active", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Fallback to background subtraction
            bg_detections = self.detect_with_background_subtraction(frame)
            detections.extend(bg_detections)
            cv2.putText(processed_frame, "Motion Detection Active", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Track vehicles
        tracked_vehicles = self.track_vehicles(detections, frame.shape)
        
        # Process tracked vehicles
        for vehicle in tracked_vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_type = vehicle['type']
            confidence = vehicle['confidence']
            status = vehicle['status']
            vehicle_id = vehicle['vehicle_id']
            
            # Count vehicles
            if vehicle_type in vehicle_counts:
                vehicle_counts[vehicle_type] += 1
            
            # Generate license plate
            license_plate = self.generate_license_plate()
            timestamp = datetime.now()
            
            # Store in database
            self.store_vehicle_log(vehicle_id, license_plate, vehicle_type, status, 
                                 timestamp, confidence, x1, y1, x2, y2)
            
            # Add to entry/exit log
            self.entry_exit_log.append({
                'vehicle_id': vehicle_id,
                'registration_number': license_plate,
                'vehicle_type': vehicle_type,
                'status': status,
                'entry_time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': timestamp.strftime('%Y-%m-%d %H:%M:%S') if status == 'Exit' else None
            })
            
            # Draw detection
            color = (0, 255, 0) if status == 'Entry' else (0, 0, 255) if status == 'Exit' else (255, 255, 0)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add labels
            cv2.putText(processed_frame, f'{vehicle_type}: {confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(processed_frame, f'ID: {vehicle_id}', 
                       (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(processed_frame, f'Plate: {license_plate}', 
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(processed_frame, f'{status}', 
                       (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Keep only recent entries
        if len(self.entry_exit_log) > 100:
            self.entry_exit_log = self.entry_exit_log[-100:]
        
        # Add frame info
        cv2.putText(processed_frame, f"Frame: {self.frame_count} | Detections: {len(tracked_vehicles)}", 
                   (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"Total Logged: {len(self.entry_exit_log)} | Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        stats = {
            'total_count': len(self.entry_exit_log),
            'vehicle_counts': vehicle_counts,
            'active_tracks': len(tracked_vehicles),
            'timestamp': datetime.now().isoformat(),
            'entry_exit_log': self.entry_exit_log[-10:]
        }
        
        return processed_frame, stats
    
    def store_vehicle_log(self, vehicle_id, registration_number, vehicle_type, status, 
                         timestamp, confidence, x1, y1, x2, y2):
        try:
            conn = sqlite3.connect('vehicle_tracking.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO vehicle_logs 
                (vehicle_id, registration_number, vehicle_type, status, entry_time, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (vehicle_id, registration_number, vehicle_type, status, timestamp, confidence,
                  x1, y1, x2, y2))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    def get_vehicle_details(self):
        try:
            conn = sqlite3.connect('vehicle_tracking.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT vehicle_id, registration_number, vehicle_type, status, entry_time, exit_time
                FROM vehicle_logs 
                ORDER BY entry_time DESC 
                LIMIT 20
            ''')
            results = cursor.fetchall()
            conn.close()
            
            vehicle_details = []
            for row in results:
                vehicle_details.append({
                    'vehicle_id': row[0],
                    'registration_number': row[1],
                    'vehicle_type': row[2],
                    'status': row[3],
                    'entry_time': row[4],
                    'exit_time': row[5] or 'Still Inside'
                })
            
            return vehicle_details
        except Exception as e:
            print(f"Database error: {e}")
            return []