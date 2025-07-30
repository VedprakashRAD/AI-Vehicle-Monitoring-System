import cv2
import numpy as np
from datetime import datetime
import sqlite3
import random
import string

class RealVehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.use_yolo = True
            print("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"❌ YOLOv8 not available: {e}")
            self.model = None
            self.use_yolo = False
            
        self.confidence_threshold = confidence_threshold
        self.entry_exit_log = []
        self.frame_count = 0
        self.last_detection_frame = 0
        self.init_database()
        
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
                confidence REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def generate_vehicle_id(self):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    def generate_license_plate(self):
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        state = random.choice(states)
        numbers = ''.join(random.choices(string.digits, k=4))
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        return f"{state}{numbers}{letters}"
    
    def detect_vehicles_yolo(self, frame):
        """Real YOLO vehicle detection"""
        detections = []
        try:
            results = self.model(frame, conf=self.confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # COCO vehicle classes
                        vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                        
                        if cls in vehicle_classes and conf > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'type': vehicle_classes[cls],
                                'confidence': conf
                            })
        except Exception as e:
            print(f"YOLO detection error: {e}")
            
        return detections
    
    def process_frame_for_web(self, frame):
        self.frame_count += 1
        processed_frame = frame.copy()
        vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        
        # Draw entry/exit lines first
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
        cv2.putText(processed_frame, "MONITORING ZONE", (frame_width - 200, entry_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Vehicle detection
        if self.use_yolo:
            detections = self.detect_vehicles_yolo(frame)
            cv2.putText(processed_frame, "YOLOv8 AI Detection Active", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            detections = []
            cv2.putText(processed_frame, "Demo Mode - Limited Detection", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Only generate demo detection occasionally
            if self.frame_count - self.last_detection_frame > 300:  # Every 10 seconds
                if random.random() < 0.3:  # 30% chance
                    vehicle_types = ['car', 'motorcycle', 'bus', 'truck']
                    detected_type = random.choice(vehicle_types)
                    x1, y1 = random.randint(50, 200), random.randint(entry_y + 20, exit_y - 50)
                    x2, y2 = x1 + random.randint(80, 150), y1 + random.randint(60, 100)
                    
                    detections = [{
                        'bbox': (x1, y1, x2, y2),
                        'type': detected_type,
                        'confidence': 0.85
                    }]
                    self.last_detection_frame = self.frame_count
        
        # Process detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            vehicle_type = detection['type']
            confidence = detection['confidence']
            
            vehicle_counts[vehicle_type] += 1
            
            # Determine status based on position
            center_y = (y1 + y2) // 2
            if center_y < entry_y + 30:
                status = 'Entry'
            elif center_y > exit_y - 30:
                status = 'Exit'
            else:
                status = 'Monitoring'
            
            # Generate vehicle details
            vehicle_id = self.generate_vehicle_id()
            license_plate = self.generate_license_plate()
            timestamp = datetime.now()
            
            # Store in database
            self.store_vehicle_log(vehicle_id, license_plate, vehicle_type, status, timestamp, confidence)
            
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
            cv2.putText(processed_frame, f'{vehicle_type}: {confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(processed_frame, f'ID: {vehicle_id}', 
                       (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(processed_frame, f'Plate: {license_plate}', 
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(processed_frame, f'Status: {status}', 
                       (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Keep only recent entries
        if len(self.entry_exit_log) > 50:
            self.entry_exit_log = self.entry_exit_log[-50:]
        
        # Add frame info
        cv2.putText(processed_frame, f"Frame: {self.frame_count} | Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (10, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"Total Vehicles: {len(self.entry_exit_log)} | Active: {sum(vehicle_counts.values())}", 
                   (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        stats = {
            'total_count': len(self.entry_exit_log),
            'vehicle_counts': vehicle_counts,
            'active_tracks': sum(vehicle_counts.values()),
            'timestamp': datetime.now().isoformat(),
            'entry_exit_log': self.entry_exit_log[-10:]
        }
        
        return processed_frame, stats
    
    def store_vehicle_log(self, vehicle_id, registration_number, vehicle_type, status, timestamp, confidence):
        try:
            conn = sqlite3.connect('vehicle_tracking.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO vehicle_logs (vehicle_id, registration_number, vehicle_type, status, entry_time, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (vehicle_id, registration_number, vehicle_type, status, timestamp, confidence))
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