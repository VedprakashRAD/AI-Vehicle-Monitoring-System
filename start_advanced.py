#!/usr/bin/env python3
"""
Advanced Vehicle Monitoring System Startup Script
=================================================

This script initializes and starts the advanced vehicle monitoring system
with all enhanced features including multi-camera support, analytics,
and real-time web dashboard.
"""

import os
import sys
import subprocess
import time
import logging
import signal
import threading
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedSystemManager:
    """Manages the startup and shutdown of the advanced vehicle monitoring system"""
    
    def __init__(self):
        self.processes = []
        self.shutdown_event = threading.Event()
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("Checking system dependencies...")
        
        required_packages = [
            'cv2', 'numpy', 'torch', 'ultralytics', 'flask', 'flask_socketio',
            'flask_restx', 'sqlite3', 'redis', 'yaml', 'psutil', 'websockets',
            'matplotlib', 'seaborn', 'pandas', 'scipy', 'scikit-learn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'ultralytics':
                    from ultralytics import YOLO
                elif package == 'flask_socketio':
                    from flask_socketio import SocketIO
                elif package == 'flask_restx':
                    from flask_restx import Api
                elif package == 'scikit-learn':
                    import sklearn
                else:
                    __import__(package)
                logger.info(f"✓ {package} is available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} is missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Installing missing packages...")
            
            # Map package names to pip install names
            pip_names = {
                'cv2': 'opencv-python',
                'flask_socketio': 'flask-socketio',
                'flask_restx': 'flask-restx',
                'scikit-learn': 'scikit-learn'
            }
            
            for package in missing_packages:
                pip_name = pip_names.get(package, package)
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
                    logger.info(f"✓ Installed {pip_name}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {pip_name}: {e}")
                    return False
        
        logger.info("All dependencies are satisfied")
        return True
    
    def check_system_requirements(self):
        """Check system requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info(f"✓ Python version: {sys.version}")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
                logger.warning("Warning: Less than 4GB RAM available. Performance may be limited.")
            else:
                logger.info(f"✓ Available RAM: {memory.total / (1024**3):.1f} GB")
        except ImportError:
            logger.warning("Cannot check memory requirements (psutil not available)")
        
        # Check camera availability
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                logger.info("✓ Default camera is available")
                cap.release()
            else:
                logger.warning("⚠ No camera detected. System will run in demo mode.")
        except ImportError:
            logger.warning("Cannot check camera availability (OpenCV not available)")
        
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        logger.info("Setting up directories...")
        
        directories = [
            'logs',
            'models',
            'data',
            'exports',
            'static/css',
            'static/js',
            'static/images'
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"✓ Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
        
        return True
    
    def download_models(self):
        """Download required AI models"""
        logger.info("Checking AI models...")
        
        try:
            from ultralytics import YOLO
            
            # Download YOLOv8 model if not present
            model_path = 'yolov8l.pt'
            if not os.path.exists(model_path):
                logger.info("Downloading YOLOv8 model...")
                model = YOLO(model_path)  # This will download the model
                logger.info("✓ YOLOv8 model downloaded successfully")
            else:
                logger.info("✓ YOLOv8 model already available")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to download models: {e}")
            return False
    
    def initialize_database(self):
        """Initialize the database"""
        logger.info("Initializing database...")
        
        try:
            from advanced_system import DatabaseManager
            db_manager = DatabaseManager()
            logger.info("✓ Database initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def start_web_dashboard(self):
        """Start the web dashboard"""
        logger.info("Starting advanced web dashboard...")
        
        try:
            # Start the web dashboard in a separate process
            process = subprocess.Popen([
                sys.executable, 'advanced_web_dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            logger.info("✓ Advanced web dashboard started successfully")
            logger.info("🌐 Dashboard available at: http://localhost:8080")
            logger.info("📊 API documentation at: http://localhost:8080/api/docs/")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start web dashboard: {e}")
            return False
    
    def start_monitoring_system(self):
        """Start the monitoring system"""
        logger.info("Starting advanced monitoring system...")
        
        try:
            # Start monitoring system with demo mode
            process = subprocess.Popen([
                sys.executable, 'advanced_system.py', '--demo'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(process)
            logger.info("✓ Advanced monitoring system started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")
            return False
    
    def print_system_info(self):
        """Print system information and instructions"""
        print("\n" + "="*80)
        print("🚗 ADVANCED VEHICLE MONITORING SYSTEM")
        print("="*80)
        print(f"🌐 Web Dashboard: http://localhost:8080")
        print(f"📊 API Documentation: http://localhost:8080/api/docs/")
        print(f"🔧 System Control: Use the web interface or API endpoints")
        print("\n📋 AVAILABLE FEATURES:")
        print("   • Multi-camera monitoring")
        print("   • Real-time vehicle detection and tracking")
        print("   • License plate recognition")
        print("   • Speed estimation and violation detection")
        print("   • Traffic analytics and heatmaps")
        print("   • System performance monitoring")
        print("   • RESTful API with comprehensive endpoints")
        print("   • WebSocket real-time updates")
        print("\n🎯 GETTING STARTED:")
        print("   1. Open the web dashboard in your browser")
        print("   2. Click 'Start Monitoring' to begin vehicle detection")
        print("   3. View live camera feeds and statistics")
        print("   4. Access analytics for traffic insights")
        print("   5. Use API endpoints for custom integrations")
        print("\n⚠️  CONTROLS:")
        print("   • Press Ctrl+C to stop the system")
        print("   • Check 'startup.log' for detailed logs")
        print("   • Modify 'advanced_config.yaml' for custom settings")
        print("="*80)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal. Stopping system...")
        self.shutdown_event.set()
        self.stop_all_processes()
        sys.exit(0)
    
    def stop_all_processes(self):
        """Stop all running processes"""
        logger.info("Stopping all processes...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info("✓ Process stopped successfully")
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't stop gracefully, forcing termination...")
                process.kill()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
    
    def run(self):
        """Main startup sequence"""
        logger.info("Starting Advanced Vehicle Monitoring System...")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Startup sequence
        startup_steps = [
            ("Checking dependencies", self.check_dependencies),
            ("Checking system requirements", self.check_system_requirements),
            ("Setting up directories", self.setup_directories),
            ("Downloading AI models", self.download_models),
            ("Initializing database", self.initialize_database),
            ("Starting web dashboard", self.start_web_dashboard),
        ]
        
        for step_name, step_function in startup_steps:
            logger.info(f"Running: {step_name}")
            if not step_function():
                logger.error(f"Failed at step: {step_name}")
                self.stop_all_processes()
                return False
            time.sleep(1)  # Brief pause between steps
        
        # Print system information
        self.print_system_info()
        
        # Keep the main process running
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
                
                # Check if processes are still running
                for process in self.processes[:]:  # Create a copy to iterate
                    if process.poll() is not None:
                        logger.warning("A process has stopped unexpectedly")
                        self.processes.remove(process)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.stop_all_processes()
        
        logger.info("Advanced Vehicle Monitoring System stopped")
        return True

def main():
    """Main entry point"""
    print("🚀 Initializing Advanced Vehicle Monitoring System...")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create and run the system manager
    manager = AdvancedSystemManager()
    success = manager.run()
    
    if success:
        print("✅ System started successfully!")
    else:
        print("❌ System startup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
