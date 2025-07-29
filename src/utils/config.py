"""
Configuration Management Module
Handles application configuration and settings.
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Application configuration manager"""
    
    def __init__(self, config_path="config/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file or use defaults"""
        default_config = {
            "database": {
                "path": "data/vehicle_counts.db",
                "backup_interval": 3600  # 1 hour
            },
            "ai_model": {
                "confidence_threshold": 0.5,
                "model_path": "models/yolo.weights",
                "classes": ["car", "motorcycle", "bus", "truck"]
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8080,
                "debug": True,
                "max_upload_size": 16777216  # 16MB
            },
            "camera": {
                "default_source": 0,
                "fps": 30,
                "resolution": [640, 480]
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/vehicle_monitor.log"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    return self._merge_configs(default_config, config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
                return default_config
        else:
            logger.info("Config file not found. Using default configuration.")
            self._save_config(default_config)
            return default_config
    
    def _merge_configs(self, default, custom):
        """Recursively merge custom config with defaults"""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def _save_config(self, config):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key, default=None):
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config(self.config)
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()


# Global configuration instance
config = Config()
