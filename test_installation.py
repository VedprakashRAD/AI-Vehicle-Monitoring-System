#!/usr/bin/env python3
"""
Test script to verify the AI Vehicle Monitoring System installation
"""

import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        ('Flask', 'flask'),
        ('Flask-SocketIO', 'flask_socketio'),
        ('OpenCV', 'cv2'),
        ('NumPy', 'numpy'),
        ('Ultralytics YOLO', 'ultralytics'),
        ('SQLite3', 'sqlite3'),
        ('DateTime', 'datetime'),
        ('Threading', 'threading'),
        ('JSON', 'json'),
        ('Pandas', 'pandas'),
    ]
    
    failed_imports = []
    
    for name, module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - FAILED: {e}")
            failed_imports.append(name)
        except Exception as e:
            print(f"⚠️  {name} - WARNING: {e}")
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All modules imported successfully!")
        return True

def test_yolo_model():
    """Test if YOLO model can be loaded"""
    print("\n🤖 Testing YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return False

def test_camera_access():
    """Test if camera can be accessed"""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera accessible! Frame shape: {frame.shape}")
            else:
                print("⚠️  Camera accessible but no frame captured")
            cap.release()
            return True
        else:
            print("⚠️  No camera found (this is okay if you plan to use video files)")
            return True
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\n🗄️  Testing database functionality...")
    
    try:
        import sqlite3
        import tempfile
        import os
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        # Test database operations
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                data TEXT
            )
        ''')
        
        # Insert test data
        cursor.execute('INSERT INTO test_table (timestamp, data) VALUES (?, ?)', 
                      ('2024-01-01 12:00:00', 'test'))
        
        # Query test data
        cursor.execute('SELECT * FROM test_table')
        result = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        # Cleanup
        os.unlink(db_path)
        
        if result:
            print("✅ Database functionality working!")
            return True
        else:
            print("❌ Database test failed - no data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_web_framework():
    """Test if Flask web framework works"""
    print("\n🌐 Testing web framework...")
    
    try:
        from flask import Flask
        from flask_socketio import SocketIO
        
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test_key'
        socketio = SocketIO(app)
        
        @app.route('/')
        def test_route():
            return "Test successful!"
        
        print("✅ Flask and SocketIO initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Web framework test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚗 AI Vehicle Monitoring System - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("YOLO Model", test_yolo_model),
        ("Camera Access", test_camera_access), 
        ("Database", test_database),
        ("Web Framework", test_web_framework),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your installation is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Run 'python web_dashboard.py' to start the web interface")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Click 'Start Monitoring' to begin vehicle detection")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure you're in the virtual environment")
        print("2. Try reinstalling failed modules with pip")
        print("3. Check if you have the required hardware (camera/webcam)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
