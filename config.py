"""
Configuration file for AI-Powered Vehicle Monitoring System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',  # YOLOv8 nano model (fastest)
    'model_variants': {
        'nano': 'yolov8n.pt',      # Fastest, least accurate
        'small': 'yolov8s.pt',     # Balanced
        'medium': 'yolov8m.pt',    # Good accuracy
        'large': 'yolov8l.pt',     # High accuracy
        'xlarge': 'yolov8x.pt'     # Highest accuracy, slowest
    },
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'max_detections': 1000,
    'device': 'auto',  # 'auto', 'cpu', 'cuda', or specific GPU id like '0'
}

# Vehicle Detection Configuration
VEHICLE_CONFIG = {
    'vehicle_classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck from COCO dataset
    'class_names': {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    },
    'tracking_distance_threshold': 100,  # Maximum distance for tracking
    'counting_line_position': 0.6,      # 60% from top of frame
    'speed_calculation': {
        'meters_per_pixel': 0.05,        # Calibration factor (needs adjustment per camera)
        'max_reasonable_speed': 200,     # km/h - cap for unrealistic speeds
        'min_tracking_points': 2         # Minimum points needed for speed calculation
    }
}

# Database Configuration
DATABASE_CONFIG = {
    'db_path': BASE_DIR / 'vehicle_counts.db',
    'backup_interval': 3600,  # seconds (1 hour)
    'max_records': 100000,    # Maximum records before cleanup
    'cleanup_older_than': 90, # days
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'default_fps': 30,
    'processing_fps': 15,        # Process every N frames for performance
    'frame_width': 640,          # Resize frame width (0 for original)
    'frame_height': 480,         # Resize frame height (0 for original)
    'video_sources': {
        'webcam_0': 0,
        'webcam_1': 1,
        'ip_camera': 'rtsp://username:password@ip:port/stream',
        'test_video': 'test_video.mp4'
    },
    'output_settings': {
        'save_output': False,
        'output_dir': BASE_DIR / 'output',
        'output_format': 'mp4v',
        'save_detections': True
    }
}

# Web Dashboard Configuration
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'secret_key': 'vehicle_monitoring_secret_key_change_in_production',
    'static_folder': BASE_DIR / 'static',
    'template_folder': BASE_DIR / 'templates',
    'max_content_length': 16 * 1024 * 1024,  # 16MB max file upload
}

# Real-time Streaming Configuration
STREAMING_CONFIG = {
    'stream_fps': 15,
    'stream_quality': 80,        # JPEG quality (1-100)
    'buffer_size': 10,           # Number of frames to buffer
    'enable_adaptive_quality': True,
    'websocket_timeout': 60,
}

# Alerting Configuration
ALERT_CONFIG = {
    'enable_alerts': True,
    'high_traffic_threshold': 50,    # vehicles per hour
    'speeding_threshold': 80,        # km/h
    'alert_cooldown': 300,           # seconds between similar alerts
    'notification_methods': {
        'email': {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'recipients': []
        },
        'webhook': {
            'enabled': False,
            'url': '',
            'timeout': 10
        }
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': {
        'enabled': True,
        'filename': BASE_DIR / 'logs' / 'vehicle_monitoring.log',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    },
    'console_logging': {
        'enabled': True
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_threads': 4,
    'memory_limit_mb': 2048,
    'enable_gpu_acceleration': True,
    'batch_processing': {
        'enabled': False,
        'batch_size': 4
    },
    'optimization': {
        'enable_tensorrt': False,    # For NVIDIA GPUs
        'enable_openvino': False,    # For Intel hardware
        'half_precision': False,     # Use FP16 instead of FP32
    }
}

# Security Configuration
SECURITY_CONFIG = {
    'enable_authentication': False,
    'api_key_required': False,
    'allowed_hosts': ['localhost', '127.0.0.1'],
    'cors_origins': '*',
    'rate_limit': {
        'enabled': False,
        'requests_per_minute': 60
    }
}

# Data Export Configuration
EXPORT_CONFIG = {
    'formats': ['csv', 'json', 'excel'],
    'max_export_records': 10000,
    'include_images': False,
    'compression': True
}

# Environment-specific overrides
def load_environment_config():
    """Load configuration overrides from environment variables"""
    config_overrides = {}
    
    # Model configuration from environment
    if os.getenv('YOLO_MODEL_PATH'):
        config_overrides['model_path'] = os.getenv('YOLO_MODEL_PATH')
    
    if os.getenv('CONFIDENCE_THRESHOLD'):
        config_overrides['confidence_threshold'] = float(os.getenv('CONFIDENCE_THRESHOLD'))
    
    # Database configuration from environment
    if os.getenv('DATABASE_PATH'):
        config_overrides['db_path'] = os.getenv('DATABASE_PATH')
    
    # Web configuration from environment
    if os.getenv('WEB_HOST'):
        config_overrides['web_host'] = os.getenv('WEB_HOST')
    
    if os.getenv('WEB_PORT'):
        config_overrides['web_port'] = int(os.getenv('WEB_PORT'))
    
    return config_overrides

# Apply environment overrides
ENV_CONFIG = load_environment_config()

# Utility function to get configuration value
def get_config(key, default=None):
    """Get configuration value with environment override support"""
    return ENV_CONFIG.get(key, default)

# Validate configuration
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if model file exists or is a valid model name
    model_path = MODEL_CONFIG['model_path']
    if not os.path.exists(model_path) and model_path not in MODEL_CONFIG['model_variants'].values():
        errors.append(f"Model file not found: {model_path}")
    
    # Check database directory exists
    db_dir = DATABASE_CONFIG['db_path'].parent
    if not db_dir.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
    
    # Check output directory
    if VIDEO_CONFIG['output_settings']['save_output']:
        output_dir = VIDEO_CONFIG['output_settings']['output_dir']
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check log directory
    if LOGGING_CONFIG['file_logging']['enabled']:
        log_dir = LOGGING_CONFIG['file_logging']['filename'].parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
    
    return errors

if __name__ == "__main__":
    # Validate configuration when run directly
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed!")
        
    # Print current configuration
    print("\nCurrent Configuration:")
    print(f"Model: {MODEL_CONFIG['model_path']}")
    print(f"Database: {DATABASE_CONFIG['db_path']}")
    print(f"Web Interface: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    print(f"Confidence Threshold: {MODEL_CONFIG['confidence_threshold']}")
    print(f"Vehicle Classes: {list(VEHICLE_CONFIG['class_names'].values())}")
