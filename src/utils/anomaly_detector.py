"""
Intelligent Anomaly Detection System
Detects unusual traffic patterns and behaviors in real-time.
"""

import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import threading
import time

logger = logging.getLogger(__name__)


class TrafficAnomalyDetector:
    """Advanced traffic anomaly detection system"""
    
    def __init__(self):
        # Historical data for pattern learning
        self.hourly_baselines = defaultdict(list)  # Hour -> [vehicle counts]
        self.speed_baselines = defaultdict(list)   # Vehicle type -> [speeds]
        self.density_history = deque(maxlen=100)   # Recent traffic density
        
        # Anomaly tracking
        self.active_anomalies = {}
        self.anomaly_history = []
        self.last_analysis_time = datetime.now()
        
        # Configuration thresholds
        self.config = {
            'congestion_threshold': 0.8,      # 80% above normal
            'stopped_vehicle_timeout': 30,    # 30 seconds stationary
            'speed_anomaly_factor': 2.0,      # 2x normal speed variation
            'density_spike_factor': 1.5,      # 1.5x normal density
            'analysis_interval': 5             # Analyze every 5 seconds
        }
        
        # Learning phase
        self.learning_mode = True
        self.learning_samples = 0
        self.min_learning_samples = 50
        
        logger.info("Traffic Anomaly Detector initialized")
    
    def update_traffic_data(self, stats):
        """Update with latest traffic statistics"""
        try:
            current_hour = datetime.now().hour
            total_vehicles = stats.get('total_count', 0)
            active_tracks = stats.get('active_tracks', 0)
            vehicle_counts = stats.get('vehicle_counts', {})
            
            # Update baselines during learning phase
            if self.learning_mode:
                self.hourly_baselines[current_hour].append(total_vehicles)
                self.density_history.append(active_tracks)
                self.learning_samples += 1
                
                if self.learning_samples >= self.min_learning_samples:
                    self.learning_mode = False
                    logger.info("Anomaly detector learning phase completed")
            
            # Analyze for anomalies
            if not self.learning_mode:
                self._analyze_anomalies(stats)
                
        except Exception as e:
            logger.error(f"Error updating traffic data: {e}")
    
    def _analyze_anomalies(self, stats):
        """Analyze current traffic for anomalies"""
        now = datetime.now()
        if (now - self.last_analysis_time).seconds < self.config['analysis_interval']:
            return
        
        self.last_analysis_time = now
        anomalies = []
        
        # Check for traffic congestion
        congestion_anomaly = self._detect_congestion(stats)
        if congestion_anomaly:
            anomalies.append(congestion_anomaly)
        
        # Check for stopped vehicles
        stopped_vehicles = self._detect_stopped_vehicles(stats)
        if stopped_vehicles:
            anomalies.extend(stopped_vehicles)
        
        # Check for speed anomalies
        speed_anomalies = self._detect_speed_anomalies(stats)
        if speed_anomalies:
            anomalies.extend(speed_anomalies)
        
        # Check for unusual density spikes
        density_anomaly = self._detect_density_spike(stats)
        if density_anomaly:
            anomalies.append(density_anomaly)
        
        # Process detected anomalies
        for anomaly in anomalies:
            self._process_anomaly(anomaly)
    
    def _detect_congestion(self, stats):
        """Detect traffic congestion based on historical patterns"""
        current_hour = datetime.now().hour
        current_count = stats.get('total_count', 0)
        
        if current_hour not in self.hourly_baselines:
            return None
        
        historical_counts = self.hourly_baselines[current_hour]
        if len(historical_counts) < 5:  # Need sufficient data
            return None
        
        avg_baseline = np.mean(historical_counts)
        threshold = avg_baseline * (1 + self.config['congestion_threshold'])
        
        if current_count > threshold:
            severity = min(100, int(((current_count - threshold) / threshold) * 100))
            return {
                'type': 'congestion',
                'severity': severity,
                'message': f'Traffic congestion detected: {current_count} vehicles (normal: {int(avg_baseline)})',
                'location': 'monitoring_zone',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'current_count': current_count,
                    'baseline_avg': avg_baseline,
                    'threshold': threshold
                }
            }
        return None
    
    def _detect_stopped_vehicles(self, stats):
        """Detect vehicles that have been stationary too long"""
        stopped_anomalies = []
        entry_exit_log = stats.get('entry_exit_log', [])
        
        for vehicle_log in entry_exit_log:
            if 'positions' in vehicle_log:
                positions = vehicle_log['positions']
                if len(positions) >= 3:
                    # Check if vehicle hasn't moved significantly
                    recent_positions = positions[-3:]
                    distances = []
                    
                    for i in range(1, len(recent_positions)):
                        pos1, pos2 = recent_positions[i-1], recent_positions[i]
                        distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                        distances.append(distance)
                    
                    avg_movement = np.mean(distances) if distances else 0
                    
                    if avg_movement < 5:  # Less than 5 pixels movement
                        stopped_anomalies.append({
                            'type': 'stopped_vehicle',
                            'severity': 60,
                            'message': f'Vehicle {vehicle_log.get("id", "unknown")} appears to be stopped',
                            'location': f'Position: {positions[-1]}',
                            'timestamp': datetime.now().isoformat(),
                            'data': {
                                'vehicle_id': vehicle_log.get('id'),
                                'vehicle_type': vehicle_log.get('vehicle_type'),
                                'avg_movement': avg_movement,
                                'position': positions[-1]
                            }
                        })
        
        return stopped_anomalies
    
    def _detect_speed_anomalies(self, stats):
        """Detect unusual vehicle speeds"""
        speed_anomalies = []
        speeds = stats.get('speeds', [])
        
        if not speeds:
            return speed_anomalies
        
        # Calculate speed statistics
        current_avg_speed = np.mean(speeds)
        current_max_speed = max(speeds)
        
        # Check for unusually high speeds
        if current_max_speed > 80:  # > 80 km/h in monitoring zone
            speed_anomalies.append({
                'type': 'high_speed',
                'severity': min(100, int((current_max_speed - 80) * 2)),
                'message': f'High speed detected: {current_max_speed:.1f} km/h',
                'location': 'monitoring_zone',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'max_speed': current_max_speed,
                    'avg_speed': current_avg_speed,
                    'speed_limit': 50  # Assumed speed limit
                }
            })
        
        # Check for unusually low average speeds (potential congestion)
        if len(speeds) > 3 and current_avg_speed < 5:  # < 5 km/h average
            speed_anomalies.append({
                'type': 'low_speed',
                'severity': 70,
                'message': f'Very low average speed: {current_avg_speed:.1f} km/h (possible congestion)',
                'location': 'monitoring_zone',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'avg_speed': current_avg_speed,
                    'vehicle_count': len(speeds)
                }
            })
        
        return speed_anomalies
    
    def _detect_density_spike(self, stats):
        """Detect sudden spikes in traffic density"""
        current_density = stats.get('active_tracks', 0)
        self.density_history.append(current_density)
        
        if len(self.density_history) < 10:
            return None
        
        recent_avg = np.mean(list(self.density_history)[-10:])
        historical_avg = np.mean(list(self.density_history)[:-10]) if len(self.density_history) > 10 else recent_avg
        
        if recent_avg > historical_avg * self.config['density_spike_factor']:
            return {
                'type': 'density_spike',
                'severity': 75,
                'message': f'Traffic density spike: {int(recent_avg)} active vehicles (normal: {int(historical_avg)})',
                'location': 'monitoring_zone',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'current_density': recent_avg,
                    'historical_avg': historical_avg,
                    'spike_factor': recent_avg / historical_avg if historical_avg > 0 else 0
                }
            }
        return None
    
    def _process_anomaly(self, anomaly):
        """Process detected anomaly"""
        anomaly_key = f"{anomaly['type']}_{anomaly['location']}"
        
        # Check if this is a new anomaly or update to existing
        if anomaly_key in self.active_anomalies:
            # Update existing anomaly
            self.active_anomalies[anomaly_key]['last_seen'] = datetime.now()
            self.active_anomalies[anomaly_key]['count'] += 1
            self.active_anomalies[anomaly_key]['severity'] = max(
                self.active_anomalies[anomaly_key]['severity'], 
                anomaly['severity']
            )
        else:
            # New anomaly
            anomaly.update({
                'id': f"anom_{int(time.time())}_{len(self.active_anomalies)}",
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'count': 1,
                'status': 'active'
            })
            self.active_anomalies[anomaly_key] = anomaly
            logger.warning(f"New anomaly detected: {anomaly['message']}")
        
        # Add to history
        self.anomaly_history.append(anomaly.copy())
        
        # Limit history size
        if len(self.anomaly_history) > 100:
            self.anomaly_history = self.anomaly_history[-100:]
    
    def get_active_anomalies(self):
        """Get currently active anomalies"""
        # Clean up old anomalies (older than 5 minutes)
        current_time = datetime.now()
        expired_keys = []
        
        for key, anomaly in self.active_anomalies.items():
            if (current_time - anomaly['last_seen']).seconds > 300:  # 5 minutes
                expired_keys.append(key)
        
        for key in expired_keys:
            self.active_anomalies[key]['status'] = 'resolved'
            del self.active_anomalies[key]
        
        return list(self.active_anomalies.values())
    
    def get_anomaly_summary(self):
        """Get summary of anomaly detection system"""
        return {
            'active_anomalies': len(self.active_anomalies),
            'total_detected': len(self.anomaly_history),
            'learning_mode': self.learning_mode,
            'learning_progress': min(100, int((self.learning_samples / self.min_learning_samples) * 100)),
            'last_analysis': self.last_analysis_time.isoformat(),
            'system_status': 'learning' if self.learning_mode else 'monitoring'
        }
    
    def get_recent_anomalies(self, limit=10):
        """Get recent anomalies"""
        return self.anomaly_history[-limit:] if self.anomaly_history else []


# Global anomaly detector instance
anomaly_detector = TrafficAnomalyDetector()
