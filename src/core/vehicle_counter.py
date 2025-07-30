"""
Vehicle Counter Module
Handles AI-based vehicle detection and counting from video streams using YOLOv8.
"""

import cv2
import numpy as np
from datetime import datetime
import logging
import uuid
import math
from collections import defaultdict, deque

# Kalman Filter for tracking
class KalmanFilter:
    def __init__(self, dt=1, u_x=0, u_y=0, std_acc=1, x_std_meas=0.1, y_std_meas=0.1):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        self.A = np.array([[1, self.dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0, 1]])
        self.B = np.array([[(self.dt**2)/2, 0], [self.dt, 0], [0, (self.dt**2)/2], [0, self.dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.array([[(self.dt**4)/4, (self.dt**3)/2, 0, 0],
                           [(self.dt**3)/2, self.dt**2, 0, 0],
                           [0, 0, (self.dt**4)/4, (self.dt**3)/2],
                           [0, 0, (self.dt**3)/2, self.dt**2]]) * std_acc**2
        self.R = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])
        self.P = np.eye(self.A.shape[1])
        self.x = np.zeros((self.A.shape[1], 1))

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)


class VehicleTracker:
    """Individual vehicle tracking class"""
    
    def __init__(self, vehicle_id, bbox, vehicle_type, confidence):
        self.id = vehicle_id
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.confidence = confidence
        self.positions = deque(maxlen=20)  # Increased for better trajectory
        self.kf = KalmanFilter()  # Initialize Kalman Filter
        self.entry_time = datetime.now()
        self.exit_time = None
        self.has_crossed_line = False
        self.speed = 0.0
        self.frames_since_update = 0
        
        # Initialize Kalman Filter with first detection
        self.kf.x[0] = bbox[0] + bbox[2] / 2
        self.kf.x[2] = bbox[1] + bbox[3] / 2
        
        # Add initial position
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        self.positions.append((center_x, center_y))
    
    def update(self, bbox, confidence):
        """Update tracker with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.frames_since_update = 0
        
        # Update Kalman Filter
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        self.kf.update(np.array([[center_x], [center_y]]))
        
        # Update position history with corrected value
        self.positions.append((self.kf.x[0,0], self.kf.x[2,0]))
        
        # Calculate speed if we have multiple positions
        if len(self.positions) >= 2:
            self._calculate_speed()
    
    def _calculate_speed(self):
        """Calculate vehicle speed based on position history"""
        if len(self.positions) < 2:
            return
        
        # Calculate distance between last two positions
        pos1 = self.positions[-2]
        pos2 = self.positions[-1]
        pixel_distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        
        # Convert pixel distance to real-world distance (rough estimation)
        # Assume 1 pixel = 0.1 meters (this should be calibrated)
        real_distance = pixel_distance * 0.1
        
        # Assume 30 FPS, so time between frames is 1/30 seconds
        time_diff = 1/30
        
        # Speed in m/s, convert to km/h
        self.speed = (real_distance / time_diff) * 3.6
    
    def is_expired(self, max_frames=30):
        """Check if tracker should be removed due to inactivity"""
        return self.frames_since_update > max_frames
    
    def predict_next_position(self):
        """Predict next position with Kalman Filter"""
        predicted_x = self.kf.predict()
        return (predicted_x[0,0], predicted_x[2,0])
    
    def increment_frames_since_update(self):
        """Increment frames since last update"""
        self.frames_since_update += 1


class VehicleCounter:
    """Advanced vehicle counter class with YOLO detection and tracking"""
    
    def __init__(self, confidence_threshold=0.5, model_path="yolov8n.pt"):
        self.confidence_threshold = confidence_threshold
        self.vehicle_counts = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0
        }
        self.total_count = 0
        self.last_update = datetime.now()
        
        # Vehicle tracking
        self.trackers = {}
        self.next_id = 1
        self.entry_exit_log = []
        
        # Counting line (horizontal line in the middle of frame)
        self.counting_line_y = None
        self.line_thickness = 3
        
        # YOLO class mapping for vehicles
        self.vehicle_classes = {
            2: 'car',      # car
            3: 'motorcycle', # motorcycle  
            5: 'bus',      # bus
            7: 'truck'     # truck
        }
        
        # Initialize YOLO model
        self.model = None
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path):
        """Initialize YOLOv8 model"""
        try:
            if YOLO is None:
                logger.error("ultralytics not available. Using dummy mode.")
                return
            
            self.model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
    
    def process_frame(self, frame):
        """Process a single frame for vehicle detection and tracking"""
        try:
            # Set counting line if not set
            if self.counting_line_y is None:
                self.counting_line_y = frame.shape[0] // 2
            
            # Run YOLO detection
            detections = self._detect_vehicles(frame)
            
            # Predict tracker positions
            for tracker in self.trackers.values():
                tracker.predict_next_position()

            # Update trackers with new detections
            self._update_trackers(detections)
            
            # Draw annotations
            annotated_frame = self._draw_annotations(frame, detections)
            
            return annotated_frame, self.get_stats()
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, self.get_stats()
    
    def _detect_vehicles(self, frame):
        """Detect vehicles in frame using YOLO"""
        if self.model is None:
            # Return dummy detection for testing
            return self._simulate_detection(frame)
        
        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter for vehicle classes only
                        if class_id in self.vehicle_classes:
                            vehicle_type = self.vehicle_classes[class_id]
                            
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': float(confidence),
                                'vehicle_type': vehicle_type,
                                'class_id': class_id
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def _simulate_detection(self, frame):
        """Simulate vehicle detection for testing when YOLO is not available"""
        # Create some dummy detections for demonstration
        h, w = frame.shape[:2]
        detections = []
        
        # Add a simulated car moving across the frame
        import time
        x = int((time.time() * 50) % w)  # Moving car simulation
        if x < w - 100:
            detections.append({
                'bbox': [x, h//2 - 50, 100, 60],
                'confidence': 0.85,
                'vehicle_type': 'car',
                'class_id': 2
            })
        
        return detections
    
    def _update_trackers(self, detections):
        """Update vehicle trackers with new detections"""
        # Mark all trackers as not updated
        for tracker in self.trackers.values():
            tracker.increment_frames_since_update()
        
        # Match detections to existing trackers
        matched_trackers = set()
        
        for detection in detections:
            bbox = detection['bbox']
            vehicle_type = detection['vehicle_type']
            confidence = detection['confidence']
            
            # Find closest tracker by predicted position
            best_tracker_id = None
            min_distance = 100  # Increased distance threshold

            center_x = detection['bbox'][0] + detection['bbox'][2] / 2
            center_y = detection['bbox'][1] + detection['bbox'][3] / 2

            for tracker_id, tracker in self.trackers.items():
                if tracker.vehicle_type != vehicle_type or tracker_id in matched_trackers:
                    continue

                predicted_pos = tracker.predict_next_position()
                distance = math.sqrt((predicted_pos[0] - center_x)**2 + (predicted_pos[1] - center_y)**2)

                if distance < min_distance:
                    min_distance = distance
                    best_tracker_id = tracker_id
            
            if best_tracker_id is not None:
                # Update existing tracker
                self.trackers[best_tracker_id].update(bbox, confidence)
                matched_trackers.add(best_tracker_id)
                
                # Check for line crossing
                self._check_line_crossing(self.trackers[best_tracker_id])
            else:
                # Create new tracker
                tracker_id = str(self.next_id)
                self.next_id += 1
                
                new_tracker = VehicleTracker(tracker_id, bbox, vehicle_type, confidence)
                self.trackers[tracker_id] = new_tracker
        
        # Remove expired trackers
        expired_trackers = [tid for tid, tracker in self.trackers.items() if tracker.is_expired()]
        for tid in expired_trackers:
            # Log exit if vehicle crossed the line
            if self.trackers[tid].has_crossed_line:
                self._log_vehicle_exit(self.trackers[tid])
            del self.trackers[tid]
    
    def _check_line_crossing(self, tracker):
        """Check if vehicle has crossed the counting line"""
        if len(tracker.positions) < 2 or tracker.has_crossed_line:
            return
        
        # Check if vehicle crossed the counting line
        pos1 = tracker.positions[-2]
        pos2 = tracker.positions[-1]
        
        # Check if line was crossed (y-coordinate crossed the counting line)
        if ((pos1[1] < self.counting_line_y < pos2[1]) or 
            (pos1[1] > self.counting_line_y > pos2[1])):
            
            tracker.has_crossed_line = True
            
            # Count the vehicle
            if tracker.vehicle_type in self.vehicle_counts:
                self.vehicle_counts[tracker.vehicle_type] += 1
                self.total_count += 1
            
            # Log entry
            self._log_vehicle_entry(tracker)
            
            logger.info(f"Vehicle {tracker.id} ({tracker.vehicle_type}) crossed counting line")
    
    def _log_vehicle_entry(self, tracker):
        """Log vehicle entry"""
        entry_log = {
            'id': tracker.id,
            'vehicle_type': tracker.vehicle_type,
            'entry_time': tracker.entry_time.isoformat(),
            'confidence': tracker.confidence,
            'speed': tracker.speed,
            'direction': 'entry'
        }
        self.entry_exit_log.append(entry_log)
    
    def _log_vehicle_exit(self, tracker):
        """Log vehicle exit"""
        tracker.exit_time = datetime.now()
        exit_log = {
            'id': tracker.id,
            'vehicle_type': tracker.vehicle_type,
            'exit_time': tracker.exit_time.isoformat(),
            'duration': (tracker.exit_time - tracker.entry_time).total_seconds(),
            'direction': 'exit'
        }
        self.entry_exit_log.append(exit_log)
    
    def _draw_annotations(self, frame, detections):
        """Draw bounding boxes, labels, and counting line on frame"""
        annotated_frame = frame.copy()
        
        # Draw counting line
        cv2.line(annotated_frame, 
                (0, self.counting_line_y), 
                (frame.shape[1], self.counting_line_y), 
                (0, 255, 255), self.line_thickness)
        cv2.putText(annotated_frame, "COUNTING LINE", 
                   (10, self.counting_line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw vehicle detections and tracks
        for tracker in self.trackers.values():
            bbox = tracker.bbox
            x, y, w, h = bbox
            
            # Choose color based on vehicle type
            colors = {
                'car': (0, 255, 0),      # Green
                'motorcycle': (255, 0, 0), # Blue
                'bus': (0, 0, 255),      # Red
                'truck': (255, 255, 0)   # Cyan
            }
            color = colors.get(tracker.vehicle_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{tracker.vehicle_type} ID:{tracker.id}"
            if tracker.speed > 0:
                label += f" {tracker.speed:.1f}km/h"
            
            cv2.putText(annotated_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw tracking trail
            if len(tracker.positions) > 1:
                points = np.array(tracker.positions, np.int32)
                cv2.polylines(annotated_frame, [points], False, color, 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
        
        # Draw statistics overlay
        stats_y = 30
        cv2.putText(annotated_frame, f"Total Count: {self.total_count}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        stats_y += 25
        cv2.putText(annotated_frame, f"Active Tracks: {len(self.trackers)}", 
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw individual counts
        for i, (vtype, count) in enumerate(self.vehicle_counts.items()):
            stats_y += 20
            cv2.putText(annotated_frame, f"{vtype.title()}: {count}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated_frame
    
    def get_stats(self):
        """Get current detection statistics"""
        return {
            'total_count': self.total_count,
            'vehicle_counts': self.vehicle_counts.copy(),
            'active_tracks': len(self.trackers),
            'timestamp': datetime.now().isoformat(),
            'confidence_threshold': self.confidence_threshold,
            'entry_exit_log': self.entry_exit_log[-10:],  # Last 10 entries
            'speeds': [t.speed for t in self.trackers.values() if t.speed > 0]
        }
    
    def get_entry_exit_details(self):
        """Get detailed entry/exit log formatted for table display"""
        formatted_log = []
        
        # Group entries and exits by vehicle ID
        vehicle_sessions = {}
        
        for log_entry in self.entry_exit_log:
            vehicle_id = log_entry['id']
            
            if vehicle_id not in vehicle_sessions:
                vehicle_sessions[vehicle_id] = {
                    'vehicle_type': log_entry['vehicle_type'],
                    'registration_number': f"VH-{vehicle_id:04d}",  # Mock registration number
                    'entry_time': None,
                    'exit_time': None,
                    'duration': None,
                    'speed': log_entry.get('speed', 0)
                }
            
            if log_entry['direction'] == 'entry':
                vehicle_sessions[vehicle_id]['entry_time'] = log_entry['entry_time']
            elif log_entry['direction'] == 'exit':
                vehicle_sessions[vehicle_id]['exit_time'] = log_entry['exit_time']
                vehicle_sessions[vehicle_id]['duration'] = log_entry.get('duration', 0)
        
        # Convert to list format for table display
        for vehicle_id, session in vehicle_sessions.items():
            formatted_log.append({
                'id': vehicle_id,
                'vehicle_type': session['vehicle_type'].title(),
                'registration_number': session['registration_number'],
                'entry_time': session['entry_time'],
                'exit_time': session['exit_time'] or 'Still Active',
                'duration': f"{session['duration']:.1f}s" if session['duration'] else 'Active',
                'speed': f"{session['speed']:.1f} km/h" if session['speed'] > 0 else 'N/A'
            })
        
        # Sort by entry time (most recent first)
        formatted_log.sort(key=lambda x: x['entry_time'] or '', reverse=True)
        
        return formatted_log[-50:]  # Return last 50 entries
    
    def reset_counts(self):
        """Reset all vehicle counts"""
        self.vehicle_counts = {k: 0 for k in self.vehicle_counts}
        self.total_count = 0
        self.trackers.clear()
        self.entry_exit_log.clear()
        self.next_id = 1
        logger.info("Vehicle counts reset")


class WebVehicleCounter(VehicleCounter):
    """Extended vehicle counter for web dashboard integration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_stats = {}
        # Import database manager
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from database.manager import DatabaseManager
            self.db = DatabaseManager()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.db = None
    
    def process_frame_for_web(self, frame):
        """Process frame and return web-compatible results"""
        processed_frame, stats = self.process_frame(frame)
        self.latest_stats = stats
        
        # Store new detections in database
        self._store_detections_in_db()
        
        return processed_frame, stats
    
    def _store_detections_in_db(self):
        """Store vehicle detections in database"""
        if not self.db:
            return
        
        try:
            # Store each active tracker detection
            for tracker in self.trackers.values():
                if hasattr(tracker, '_stored_in_db') and tracker._stored_in_db:
                    continue  # Already stored
                
                # Mark as stored to avoid duplicates
                tracker._stored_in_db = True
                
                # Store in database
                self.db.insert_detection(
                    vehicle_type=tracker.vehicle_type,
                    confidence=tracker.confidence,
                    bbox=tracker.bbox,
                    speed=tracker.speed if tracker.speed > 0 else None,
                    metadata={
                        'tracker_id': tracker.id,
                        'positions': list(tracker.positions)[-5:],  # Last 5 positions
                        'has_crossed_line': tracker.has_crossed_line
                    }
                )
        except Exception as e:
            logger.error(f"Error storing detections in database: {e}")
