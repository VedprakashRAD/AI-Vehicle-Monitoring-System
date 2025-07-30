"""
System Monitoring Module
Provides real-time system performance metrics for the vehicle monitoring system.
"""

import psutil
import time
import threading
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SystemMonitor:
    """System performance monitoring class"""
    
    def __init__(self):
        self.cpu_percent = 0
        self.memory_percent = 0
        self.memory_used_mb = 0
        self.memory_total_mb = 0
        self.disk_usage = 0
        self.network_io = {'bytes_sent': 0, 'bytes_recv': 0}
        self.process_info = {}
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                self.cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_percent = memory.percent
                self.memory_used_mb = memory.used // (1024 * 1024)
                self.memory_total_mb = memory.total // (1024 * 1024)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage = (disk.used / disk.total) * 100
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.network_io = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                }
                
                # Current process info
                current_process = psutil.Process()
                self.process_info = {
                    'cpu_percent': current_process.cpu_percent(),
                    'memory_mb': current_process.memory_info().rss // (1024 * 1024),
                    'threads': current_process.num_threads(),
                    'connections': len(current_process.connections())
                }
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
            
            time.sleep(2)  # Update every 2 seconds
    
    def get_metrics(self):
        """Get current system metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': round(self.cpu_percent, 1),
            'memory_percent': round(self.memory_percent, 1),
            'memory_used_mb': self.memory_used_mb,
            'memory_total_mb': self.memory_total_mb,
            'disk_usage_percent': round(self.disk_usage, 1),
            'network_io': self.network_io,
            'process_info': self.process_info,
            'uptime_seconds': time.time() - psutil.boot_time()
        }
    
    def get_gpu_info(self):
        """Get GPU information if available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                return {
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
        except ImportError:
            pass
        return None
    
    def get_advanced_metrics(self):
        """Get advanced system metrics"""
        try:
            # CPU frequencies
            cpu_freq = psutil.cpu_freq()
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'current_freq': cpu_freq.current if cpu_freq else 0,
                'min_freq': cpu_freq.min if cpu_freq else 0,
                'max_freq': cpu_freq.max if cpu_freq else 0
            }
            
            # Temperature (if available)
            temperatures = {}
            try:
                temps = psutil.sensors_temperatures()
                for name, entries in temps.items():
                    temperatures[name] = [entry.current for entry in entries]
            except:
                pass
            
            return {
                'cpu_info': cpu_info,
                'temperatures': temperatures,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"Error getting advanced metrics: {e}")
            return {}


# Global system monitor instance
system_monitor = SystemMonitor()
