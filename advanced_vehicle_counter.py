#!/usr/bin/env python3
"""
Advanced Vehicle Counter for AI-Powered Monitoring System
"""

import cv2
import numpy as np
from datetime import datetime
import logging
from collections import deque
import math

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class AdvancedVehicleCounter:
    """Advanced vehicle counter with enhanced tracking and analytics"""
    
    def __init__(self, config):
        self.config = config
        self.model_config = config.get('model', {})
        self.video_config = config.get('video', {})
        self.tracker_config = config.get('tracker', {})
        
        self.confidence_threshold = self.model_config.get('confidence_threshold', 0.4)
        self.vehicle_classes = self.model_config.get('classes', {})
        
        self.trackers = {}
        self.next_id = 1
        self.entry_exit_log = []
        
        self.counting_line_y = self.video_config.get('counting_line_y', 500)
        self.line_thickness = self.video_config.get('line_thickness', 3)
        
        # Initialize YOLO model
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the YOLO model based on configuration"""
        if YOLO is None:
            logger.error("ultralytics library not found. Please install it.")
            return
        
        model_name = self.model_config.get('name', 'yolov8n.pt')
        try:
            self.model = YOLO(model_name)
            logger.info(f"Successfully loaded YOLO model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")

    def process_frame(self, frame):
        """Process a single frame to detect and track vehicles"""
        if self.model is None:
            return frame, {}
        
        try:
            # Resize frame for processing
            frame_height, frame_width = frame.shape[:2]
            proc_width = self.video_config.get('frame_width', 1280)
            proc_height = int(frame_height * (proc_width / frame_width))
            resized_frame = cv2.resize(frame, (proc_width, proc_height))
            
            # Detect vehicles
            detections = self._detect_vehicles(resized_frame)
            
            # Update trackers
            self._update_trackers(detections)
            
            # Draw annotations
            annotated_frame = self._draw_annotations(resized_frame)
            
            # Return stats
            return annotated_frame, self.get_stats()
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            return frame, {}

    def _detect_vehicles(self, frame):
        """Detect vehicles in a frame using YOLO model"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                if class_id in self.vehicle_classes:
                    vehicle_type = self.vehicle_classes[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'confidence': confidence,
                        'vehicle_type': vehicle_type
                    })
        return detections

    def _update_trackers(self, detections):
        """Update vehicle trackers with new detections"""
        # ... (Advanced tracking logic will be added here)
        pass

    def _draw_annotations(self, frame):
        """Draw annotations on the frame"""
        # ... (Advanced annotation drawing will be added here)
        return frame

    def get_stats(self):
        """Get current statistics"""
        # ... (Advanced stats calculation will be added here)
        return {}

