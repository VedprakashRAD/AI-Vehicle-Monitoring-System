"""
Database Manager for Vehicle Monitoring System
Handles all database operations including data storage, retrieval, and analytics
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from config import DATABASE_CONFIG

# Setup logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for vehicle monitoring system"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize Database Manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DATABASE_CONFIG['db_path']
        self.ensure_database_exists()
        self.setup_tables()
        
    def ensure_database_exists(self):
        """Ensure database file and directory exist"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def setup_tables(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Vehicle counts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                vehicle_type TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                location TEXT DEFAULT 'Main Road',
                speed REAL,
                confidence REAL,
                track_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Hourly summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour DATETIME NOT NULL,
                total_vehicles INTEGER DEFAULT 0,
                cars INTEGER DEFAULT 0,
                motorcycles INTEGER DEFAULT 0,
                buses INTEGER DEFAULT 0,
                trucks INTEGER DEFAULT 0,
                avg_speed REAL DEFAULT 0,
                peak_hour_traffic INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(hour)
            )
        ''')
        
        # Daily summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_vehicles INTEGER DEFAULT 0,
                cars INTEGER DEFAULT 0,
                motorcycles INTEGER DEFAULT 0,
                buses INTEGER DEFAULT 0,
                trucks INTEGER DEFAULT 0,
                avg_speed REAL DEFAULT 0,
                peak_hour TIME,
                peak_hour_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        ''')
        
        # System alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT DEFAULT 'INFO',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT FALSE,
                metadata TEXT  -- JSON string for additional data
            )
        ''')
        
        # Camera/source configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_name TEXT NOT NULL UNIQUE,
                source_path TEXT NOT NULL,
                location TEXT,
                calibration_data TEXT,  -- JSON string
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_counts_timestamp ON vehicle_counts(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_counts_type ON vehicle_counts(vehicle_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hourly_summary_hour ON hourly_summary(hour)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info("Database tables created/verified successfully")
    
    def insert_vehicle_count(self, vehicle_type: str, speed: float = None, 
                           confidence: float = None, track_id: int = None,
                           location: str = "Main Road") -> int:
        """
        Insert a vehicle count record
        
        Args:
            vehicle_type: Type of vehicle (car, motorcycle, bus, truck)
            speed: Vehicle speed in km/h
            confidence: Detection confidence score
            track_id: Vehicle tracking ID
            location: Detection location
            
        Returns:
            ID of inserted record
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO vehicle_counts 
            (timestamp, vehicle_type, speed, confidence, track_id, location)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), vehicle_type, speed, confidence, track_id, location))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.debug(f"Inserted vehicle count: {vehicle_type} (ID: {record_id})")
        return record_id
    
    def get_hourly_counts(self, hours: int = 24) -> List[Dict]:
        """
        Get vehicle counts grouped by hour
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of hourly count dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                vehicle_type,
                COUNT(*) as count,
                AVG(speed) as avg_speed,
                AVG(confidence) as avg_confidence
            FROM vehicle_counts 
            WHERE timestamp >= ?
            GROUP BY hour, vehicle_type
            ORDER BY hour DESC
        ''', (start_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Group by hour
        hourly_data = {}
        for row in results:
            hour = row['hour']
            if hour not in hourly_data:
                hourly_data[hour] = {
                    'hour': hour,
                    'total': 0,
                    'vehicles': {},
                    'avg_speed': 0,
                    'avg_confidence': 0
                }
            
            hourly_data[hour]['vehicles'][row['vehicle_type']] = row['count']
            hourly_data[hour]['total'] += row['count']
            hourly_data[hour]['avg_speed'] = row['avg_speed'] or 0
            hourly_data[hour]['avg_confidence'] = row['avg_confidence'] or 0
        
        return list(hourly_data.values())
    
    def get_daily_summary(self, days: int = 7) -> List[Dict]:
        """
        Get daily summary statistics
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of daily summary dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_vehicles,
                SUM(CASE WHEN vehicle_type = 'car' THEN 1 ELSE 0 END) as cars,
                SUM(CASE WHEN vehicle_type = 'motorcycle' THEN 1 ELSE 0 END) as motorcycles,
                SUM(CASE WHEN vehicle_type = 'bus' THEN 1 ELSE 0 END) as buses,
                SUM(CASE WHEN vehicle_type = 'truck' THEN 1 ELSE 0 END) as trucks,
                AVG(speed) as avg_speed,
                MAX(COUNT(*)) OVER (PARTITION BY DATE(timestamp)) as peak_count
            FROM vehicle_counts 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', (start_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def get_vehicle_speeds(self, vehicle_type: str = None, hours: int = 24) -> List[float]:
        """
        Get vehicle speeds for analysis
        
        Args:
            vehicle_type: Filter by vehicle type
            hours: Number of hours to look back
            
        Returns:
            List of speed values
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        if vehicle_type:
            cursor.execute('''
                SELECT speed FROM vehicle_counts 
                WHERE timestamp >= ? AND vehicle_type = ? AND speed IS NOT NULL
                ORDER BY timestamp DESC
            ''', (start_time, vehicle_type))
        else:
            cursor.execute('''
                SELECT speed FROM vehicle_counts 
                WHERE timestamp >= ? AND speed IS NOT NULL
                ORDER BY timestamp DESC
            ''', (start_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [row['speed'] for row in results]
    
    def get_traffic_patterns(self, days: int = 30) -> Dict:
        """
        Analyze traffic patterns
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with traffic pattern analytics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Hourly patterns
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count
            FROM vehicle_counts 
            WHERE timestamp >= ?
            GROUP BY hour
            ORDER BY hour
        ''', (start_date,))
        
        hourly_pattern = {f"{int(row['hour']):02d}:00": row['count'] 
                         for row in cursor.fetchall()}
        
        # Daily patterns (day of week)
        cursor.execute('''
            SELECT 
                strftime('%w', timestamp) as day_of_week,
                COUNT(*) as count
            FROM vehicle_counts 
            WHERE timestamp >= ?
            GROUP BY day_of_week
            ORDER BY day_of_week
        ''', (start_date,))
        
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 
                    'Thursday', 'Friday', 'Saturday']
        daily_pattern = {day_names[int(row['day_of_week'])]: row['count'] 
                        for row in cursor.fetchall()}
        
        # Vehicle type distribution
        cursor.execute('''
            SELECT vehicle_type, COUNT(*) as count
            FROM vehicle_counts 
            WHERE timestamp >= ?
            GROUP BY vehicle_type
            ORDER BY count DESC
        ''', (start_date,))
        
        vehicle_distribution = {row['vehicle_type']: row['count'] 
                              for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'hourly_pattern': hourly_pattern,
            'daily_pattern': daily_pattern,
            'vehicle_distribution': vehicle_distribution,
            'analysis_period_days': days
        }
    
    def insert_alert(self, alert_type: str, message: str, 
                    severity: str = 'INFO', metadata: Dict = None):
        """
        Insert a system alert
        
        Args:
            alert_type: Type of alert (traffic, speed, system, etc.)
            message: Alert message
            severity: Alert severity (INFO, WARNING, ERROR, CRITICAL)
            metadata: Additional alert data
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, message, severity, metadata)
            VALUES (?, ?, ?, ?)
        ''', (alert_type, message, severity, metadata_json))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Alert inserted: {alert_type} - {message}")
    
    def get_recent_alerts(self, hours: int = 24, acknowledged: bool = None) -> List[Dict]:
        """
        Get recent alerts
        
        Args:
            hours: Number of hours to look back
            acknowledged: Filter by acknowledgment status
            
        Returns:
            List of alert dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        query = '''
            SELECT * FROM alerts 
            WHERE timestamp >= ?
        '''
        params = [start_time]
        
        if acknowledged is not None:
            query += ' AND acknowledged = ?'
            params.append(acknowledged)
        
        query += ' ORDER BY timestamp DESC'
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in results:
            alert = dict(row)
            if alert['metadata']:
                alert['metadata'] = json.loads(alert['metadata'])
            alerts.append(alert)
        
        return alerts
    
    def export_data(self, start_date: datetime = None, end_date: datetime = None,
                   format: str = 'csv') -> str:
        """
        Export data to file
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            format: Export format ('csv', 'json', 'excel')
            
        Returns:
            Path to exported file
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        conn = self.get_connection()
        
        # Get vehicle counts data
        df = pd.read_sql_query('''
            SELECT * FROM vehicle_counts 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', conn, params=(start_date, end_date))
        
        conn.close()
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'csv':
            filename = f'vehicle_data_{timestamp}.csv'
            df.to_csv(filename, index=False)
        elif format.lower() == 'json':
            filename = f'vehicle_data_{timestamp}.json'
            df.to_json(filename, orient='records', date_format='iso')
        elif format.lower() == 'excel':
            filename = f'vehicle_data_{timestamp}.xlsx'
            df.to_excel(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data exported to: {filename}")
        return filename
    
    def cleanup_old_data(self, days: int = None):
        """
        Clean up old data from database
        
        Args:
            days: Number of days to keep (default from config)
        """
        days = days or DATABASE_CONFIG['cleanup_older_than']
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Delete old vehicle counts
        cursor.execute('DELETE FROM vehicle_counts WHERE timestamp < ?', (cutoff_date,))
        deleted_counts = cursor.rowcount
        
        # Delete old alerts
        cursor.execute('DELETE FROM alerts WHERE timestamp < ? AND acknowledged = TRUE', 
                      (cutoff_date,))
        deleted_alerts = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_counts} old vehicle records and {deleted_alerts} old alerts")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Table record counts
        tables = ['vehicle_counts', 'hourly_summary', 'daily_summary', 'alerts', 'camera_config']
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) as count FROM {table}')
            stats[f'{table}_count'] = cursor.fetchone()['count']
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        stats['database_size_bytes'] = cursor.fetchone()['size']
        
        # Date ranges
        cursor.execute('SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM vehicle_counts')
        date_range = cursor.fetchone()
        stats['data_date_range'] = {
            'first_record': date_range['first'],
            'last_record': date_range['last']
        }
        
        conn.close()
        return stats

# Utility functions
def create_sample_data(db_manager: DatabaseManager, days: int = 7):
    """Create sample data for testing"""
    import random
    
    vehicle_types = ['car', 'motorcycle', 'bus', 'truck']
    
    start_date = datetime.now() - timedelta(days=days)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        # Generate different traffic patterns for different hours
        for hour in range(24):
            # More traffic during rush hours
            if hour in [7, 8, 17, 18]:
                vehicle_count = random.randint(15, 30)
            elif hour in [9, 10, 11, 12, 13, 14, 15, 16]:
                vehicle_count = random.randint(8, 15)
            else:
                vehicle_count = random.randint(2, 8)
            
            for _ in range(vehicle_count):
                vehicle_type = random.choices(
                    vehicle_types, 
                    weights=[70, 20, 5, 5]  # Cars are most common
                )[0]
                
                speed = random.normalvariate(45, 10)  # Average 45 km/h
                speed = max(10, min(100, speed))  # Clamp between 10-100
                
                confidence = random.uniform(0.7, 0.95)
                
                # Random timestamp within the hour
                timestamp = current_date.replace(hour=hour) + timedelta(
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                
                # Insert with custom timestamp
                conn = db_manager.get_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO vehicle_counts 
                    (timestamp, vehicle_type, speed, confidence, location)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, vehicle_type, speed, confidence, 'Main Road'))
                conn.commit()
                conn.close()
    
    logger.info(f"Created {days} days of sample data")

if __name__ == "__main__":
    # Test the database manager
    db_manager = DatabaseManager()
    
    # Create some sample data for testing
    print("Creating sample data...")
    create_sample_data(db_manager, days=7)
    
    # Test various functions
    print("\nDatabase Statistics:")
    stats = db_manager.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTraffic Patterns:")
    patterns = db_manager.get_traffic_patterns(days=7)
    print(f"  Hourly pattern: {patterns['hourly_pattern']}")
    print(f"  Vehicle distribution: {patterns['vehicle_distribution']}")
    
    print("\nHourly Counts (last 24 hours):")
    hourly = db_manager.get_hourly_counts(hours=24)
    for hour_data in hourly[:5]:  # Show first 5
        print(f"  {hour_data['hour']}: {hour_data['total']} vehicles")
    
    print("\nExporting data...")
    export_file = db_manager.export_data(format='csv')
    print(f"  Exported to: {export_file}")
    
    print("\nDatabase operations completed successfully!")
