import cv2
import numpy as np
from datetime import datetime
import sqlite3
import random
import string

class EnhancedVehicleCounter:
    def __init__(self, confidence_threshold=0.5):
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.use_ai = True
        except:
            self.model = None
            self.use_ai = False
            
        self.confidence_threshold = confidence_threshold
        self.vehicle_tracks = {}
        self.entry_exit_log = []
        self.vehicle_count = 0
        self.frame_count = 0
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
                entry_time DATETIME,
                exit_time DATETIME,
                confidence REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def generate_vehicle_id(self):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    def extract_license_plate(self, vehicle_crop=None):
        # Generate realistic demo license plate
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        state = random.choice(states)
        numbers = ''.join(random.choices(string.digits, k=4))
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        return f"{state}{numbers}{letters}"
    
    def process_frame_for_web(self, frame):
        self.frame_count += 1
        processed_frame = frame.copy()
        vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        
        # Try to use YOLO if available
        if self.use_ai and self.model is not None:
            try:
                results = self.model(frame, conf=self.confidence_threshold)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # COCO dataset vehicle classes
                            vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                            
                            if cls in vehicle_classes and conf > self.confidence_threshold:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                vehicle_type = vehicle_classes[cls]
                                vehicle_counts[vehicle_type] += 1
                                
                                # Process detected vehicle
                                vehicle_id = self.generate_vehicle_id()
                                license_plate = self.extract_license_plate()
                                entry_time = datetime.now()
                                
                                # Store in database
                                self.store_vehicle_log(vehicle_id, license_plate, vehicle_type, entry_time, conf)
                                
                                # Determine entry/exit based on position
                                center_y = (y1 + y2) // 2
                                frame_height = frame.shape[0]
                                entry_line = frame_height // 3
                                exit_line = (frame_height * 2) // 3
                                
                                status = 'Entry' if center_y < entry_line + 50 else 'Exit' if center_y > exit_line - 50 else 'Monitoring'
                                
                                # Add to entry/exit log
                                self.entry_exit_log.append({
                                    'vehicle_id': vehicle_id,
                                    'registration_number': license_plate,
                                    'vehicle_type': vehicle_type,
                                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'exit_time': entry_time.strftime('%Y-%m-%d %H:%M:%S') if status == 'Exit' else None,
                                    'status': status
                                })
                                
                                # Draw detection box
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(processed_frame, f'{vehicle_type}: {conf:.2f}', 
                                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(processed_frame, f'ID: {vehicle_id}', 
                                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                cv2.putText(processed_frame, f'Plate: {license_plate}', 
                                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Add AI status overlay
                cv2.putText(processed_frame, "YOLOv8 AI Detection Active", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
            except Exception as e:
                print(f"YOLO detection error: {e}")
                self.use_ai = False
        
        # Fallback to demo mode if AI not available
        if not self.use_ai:
            cv2.putText(processed_frame, "Demo Mode - Simulated Detection", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Generate demo detection every 180 frames (6 seconds) - less frequent for better accuracy
            if self.frame_count % 180 == 0:
                vehicle_types = ['car', 'motorcycle', 'bus', 'truck']
                detected_type = random.choice(vehicle_types)
                vehicle_counts[detected_type] = 1
                
                # Generate vehicle details
                vehicle_id = self.generate_vehicle_id()
                license_plate = self.extract_license_plate()
                entry_time = datetime.now()
                
                # Store in database
                self.store_vehicle_log(vehicle_id, license_plate, detected_type, entry_time, 0.85)
                
                # Determine entry/exit for demo
                demo_status = random.choice(['Entry', 'Exit', 'Monitoring'])
                
                # Add to entry/exit log
                self.entry_exit_log.append({
                    'vehicle_id': vehicle_id,
                    'registration_number': license_plate,
                    'vehicle_type': detected_type,
                    'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': entry_time.strftime('%Y-%m-%d %H:%M:%S') if demo_status == 'Exit' else None,
                    'status': demo_status
                })
                
                # Draw demo detection box
                x1, y1, x2, y2 = random.randint(50, 200), random.randint(100, 200), random.randint(250, 400), random.randint(250, 350)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f'{detected_type}: 0.85', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(processed_frame, f'ID: {vehicle_id}', 
                           (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(processed_frame, f'Plate: {license_plate}', 
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw entry/exit lines
        frame_height, frame_width = processed_frame.shape[:2]
        
        # Entry line (top third of frame) - Green
        entry_y = frame_height // 3
        cv2.line(processed_frame, (0, entry_y), (frame_width, entry_y), (0, 255, 0), 3)
        cv2.putText(processed_frame, "ENTRY LINE", (10, entry_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Exit line (bottom third of frame) - Red
        exit_y = (frame_height * 2) // 3
        cv2.line(processed_frame, (0, exit_y), (frame_width, exit_y), (0, 0, 255), 3)
        cv2.putText(processed_frame, "EXIT LINE", (10, exit_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add monitoring zone indicator
        cv2.rectangle(processed_frame, (10, entry_y + 10), (frame_width - 10, exit_y - 10), 
                     (255, 255, 0), 2)
        cv2.putText(processed_frame, "MONITORING ZONE", (frame_width - 200, entry_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add frame info and statistics
        cv2.putText(processed_frame, f"Frame: {self.frame_count} | Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, f"Total Vehicles: {len(self.entry_exit_log)} | Active: {sum(vehicle_counts.values())}", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Keep only recent entries
        if len(self.entry_exit_log) > 50:
            self.entry_exit_log = self.entry_exit_log[-50:]
        
        stats = {
            'total_count': len(self.entry_exit_log),
            'vehicle_counts': vehicle_counts,
            'active_tracks': sum(vehicle_counts.values()),
            'timestamp': datetime.now().isoformat(),
            'entry_exit_log': self.entry_exit_log[-10:]
        }
        
        return processed_frame, stats
    
    def store_vehicle_log(self, vehicle_id, registration_number, vehicle_type, entry_time, confidence):
        try:
            conn = sqlite3.connect('vehicle_tracking.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO vehicle_logs (vehicle_id, registration_number, vehicle_type, entry_time, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (vehicle_id, registration_number, vehicle_type, entry_time, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    def get_vehicle_details(self):
        try:
            conn = sqlite3.connect('vehicle_tracking.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT vehicle_id, registration_number, vehicle_type, entry_time, exit_time
                FROM vehicle_logs 
                ORDER BY entry_time DESC 
                LIMIT 20
            ''')
            results = cursor.fetchall()
            conn.close()
            
            vehicle_details = []
            for row in results:
                # Get status from entry_exit_log if available
                status = 'Entry'  # Default status
                for log_entry in self.entry_exit_log:
                    if log_entry['vehicle_id'] == row[0]:
                        status = log_entry['status']
                        break
                
                vehicle_details.append({
                    'vehicle_id': row[0],
                    'registration_number': row[1],
                    'vehicle_type': row[2],
                    'status': status,
                    'entry_time': row[3],
                    'exit_time': row[4] or 'Still Inside'
                })
            
            return vehicle_details
        except Exception as e:
            print(f"Database error: {e}")
            return []