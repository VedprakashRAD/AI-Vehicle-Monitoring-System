#!/usr/bin/env python3
"""
Advanced AI-Powered Vehicle Monitoring System
"""

import yaml
import logging
import argparse
from src.web.dashboard import VehicleDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_config(path="advanced_config.yaml"):
    """Load advanced configuration from YAML file"""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def main():
    """Main function to start the advanced system"""
    parser = argparse.ArgumentParser(description='Advanced AI-Powered Vehicle Monitoring System')
    parser.add_argument('--config', default='advanced_config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Exiting due to configuration error")
        return
        
    try:
        print("🚗 Advanced AI-Powered Vehicle Monitoring System")
        print("=" * 60)
        print(f"Loaded configuration: {args.config}")
        print(f"AI Model: {config.get('model', {}).get('name', 'N/A')}")
        print(f"Database: {config.get('database', {}).get('path', 'N/A')}")
        print(f"Dashboard: http://{config.get('app', {}).get('host', 'N/A')}:{config.get('app', {}).get('port', 'N/A')}")
        print("=" * 60)
        
        # Initialize dashboard with advanced settings
        app_settings = config.get('app', {})
        dashboard = VehicleDashboard(
            host=app_settings.get('host', '127.0.0.1'),
            port=app_settings.get('port', 8081),
            debug=app_settings.get('debug', False),
            config=config
        )
        
        print("\n🚀 Starting advanced dashboard...")
        print("🔄 Press Ctrl+C to stop")
        dashboard.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped")
    except Exception as e:
        logger.error(f"Error starting advanced system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
