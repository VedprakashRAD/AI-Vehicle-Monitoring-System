# 🚗 Advanced AI-Powered Vehicle Monitoring System

## Overview

This is a state-of-the-art vehicle monitoring and counting system that uses artificial intelligence and computer vision to provide comprehensive traffic analysis and monitoring capabilities. The system has been significantly enhanced with advanced features including multi-camera support, real-time analytics, license plate recognition, and traffic violation detection.

## 🌟 Key Features

### Core Functionality
- **Multi-Camera Support**: Monitor multiple CCTV streams simultaneously
- **Real-Time Vehicle Detection**: Advanced YOLOv8-based vehicle detection
- **Object Tracking**: DeepSORT-like tracking with persistent vehicle IDs
- **License Plate Recognition**: CRAFT + OCR for license plate detection
- **Speed Estimation**: Real-time vehicle speed calculation
- **Traffic Violation Detection**: Automatic detection of speeding and wrong direction

### Advanced Analytics
- **Traffic Density Heatmaps**: Visual representation of traffic patterns
- **Historical Data Analysis**: Comprehensive traffic analytics over time
- **Peak Hour Detection**: Automatic identification of traffic peak hours
- **Vehicle Classification**: Detailed categorization (cars, motorcycles, buses, trucks)
- **Trajectory Analysis**: Vehicle path tracking and analysis

### Web Dashboard & API
- **Modern Web Interface**: Responsive dashboard with real-time updates
- **RESTful API**: Comprehensive API with OpenAPI documentation
- **WebSocket Integration**: Real-time data streaming
- **Multi-Zone Monitoring**: Configurable monitoring regions
- **Alert System**: Real-time notifications for violations and incidents

### System Monitoring
- **Performance Metrics**: CPU, memory, and GPU usage monitoring
- **Health Checks**: System status and service availability
- **Logging**: Comprehensive logging system
- **Database Optimization**: Time-series data storage with SQLite

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU support (optional but recommended)
- Camera or RTSP stream access

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI-Powered\ Real-Time\ Vehicle\ Monitoring\ and\ Counting\ System\ Using\ CCTV\ Feeds
   ```

2. **Run the advanced startup script**:
   ```bash
   python3 start_advanced.py
   ```

The startup script will:
- Check and install all dependencies
- Download required AI models
- Initialize the database
- Start the web dashboard
- Display access information

### Manual Installation

If you prefer manual installation:

```bash
# Install core dependencies
pip3 install opencv-python ultralytics torch torchvision
pip3 install flask flask-socketio flask-restx
pip3 install numpy pandas matplotlib seaborn
pip3 install redis pyyaml psutil websockets
pip3 install scipy scikit-learn
```

## 🚀 Usage

### Starting the System

**Option 1: Automated Start (Recommended)**
```bash
python3 start_advanced.py
```

**Option 2: Component-wise Start**
```bash
# Start the web dashboard
python3 advanced_web_dashboard.py

# In another terminal, start monitoring
python3 advanced_system.py --demo
```

### Accessing the System

Once started, you can access:
- **Web Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8080/api/docs/
- **Live Video Feed**: http://localhost:8080/video_feed/main_camera

### Configuration

Edit `advanced_config.yaml` to customize:

```yaml
cameras:
  - id: cam_001
    name: Main Entrance
    source: 0  # or RTSP URL
    resolution: [1920, 1080]
    fps: 30
    zones:
      speed_zone_1:
        - [100, 100]
        - [500, 100]
        - [500, 400]
        - [100, 400]

models:
  yolo_model: yolov8l.pt
  confidence_threshold: 0.3
  iou_threshold: 0.5

tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

analytics:
  enable_heatmaps: true
  enable_trajectory_analysis: true
  save_raw_detections: true

alerts:
  enable_speed_alerts: true
  enable_violation_alerts: true
  webhook_url: null
```

## 📊 API Endpoints

### Camera Management
- `GET /api/cameras/` - List all cameras
- `POST /api/cameras/{camera_id}/start` - Start camera monitoring
- `POST /api/cameras/{camera_id}/stop` - Stop camera monitoring

### Analytics
- `GET /api/analytics/stats` - Get traffic statistics
- `GET /api/analytics/heatmap/{camera_id}` - Get traffic heatmap
- `GET /api/analytics/violations` - Get violation statistics

### System Monitoring
- `GET /api/system/status` - Get system performance metrics
- `GET /api/system/health` - System health check

### Alerts
- `GET /api/alerts/` - Get recent alerts
- `POST /api/alerts/acknowledge/{alert_id}` - Acknowledge alert

## 🎯 Features in Detail

### 1. Multi-Camera Monitoring
The system supports multiple cameras with individual configurations:
- RTSP streams
- USB cameras
- IP cameras
- Different resolutions and frame rates per camera

### 2. Advanced Vehicle Detection
Using YOLOv8 for state-of-the-art detection:
- High accuracy vehicle detection
- Multiple vehicle types (car, motorcycle, bus, truck)
- Confidence-based filtering
- Real-time processing

### 3. Object Tracking
DeepSORT-inspired tracking system:
- Persistent vehicle IDs across frames
- Kalman filter-based prediction
- Re-identification capabilities
- Trajectory recording

### 4. License Plate Recognition
Two-stage recognition system:
- CRAFT text detection for plate localization
- OCR for character recognition
- Preprocessing for better accuracy
- Multiple plate formats support

### 5. Speed Estimation
Perspective correction-based speed calculation:
- Configurable reference points
- Real-time speed measurement
- Speed violation detection
- Historical speed analytics

### 6. Traffic Analytics
Comprehensive analytics dashboard:
- Real-time traffic counts
- Historical trend analysis
- Peak hour identification
- Vehicle type distribution
- Traffic density heatmaps

### 7. Alert System
Real-time notification system:
- Speed violations
- Wrong direction detection
- Traffic congestion alerts
- System health alerts
- Webhook integration

## 📱 Web Dashboard

The web dashboard provides:

### Main Dashboard
- Live camera feeds with annotations
- Real-time statistics cards
- Traffic flow charts
- System performance metrics
- Recent alerts panel

### Analytics Page
- Historical data visualization
- Traffic pattern analysis
- Heatmap generation
- Export capabilities
- Custom date range selection

### Camera Management
- Individual camera controls
- Configuration management
- Status monitoring
- Feed quality settings

## 🔧 Advanced Configuration

### Camera Setup
For RTSP cameras:
```yaml
cameras:
  - id: parking_lot
    name: Parking Lot Camera
    source: rtsp://username:password@192.168.1.100:554/stream
    resolution: [1920, 1080]
    fps: 25
```

### Zone Configuration
Define monitoring zones:
```yaml
zones:
  entry_zone:
    - [0, 400]
    - [800, 400]
    - [800, 600]
    - [0, 600]
  exit_zone:
    - [800, 200]
    - [1200, 200]
    - [1200, 400]
    - [800, 400]
```

### Speed Limits
Configure speed limits per zone:
```yaml
speed_limits:
  entry_zone: 25.0  # km/h
  main_road: 50.0
  parking_area: 15.0
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Verify camera index (0, 1, 2, etc.)
   - Test with external camera app

2. **Performance issues**:
   - Reduce camera resolution
   - Lower confidence threshold
   - Enable GPU acceleration
   - Close unnecessary applications

3. **Web dashboard not accessible**:
   - Check firewall settings
   - Verify port 8080 is available
   - Check system logs

4. **Installation errors**:
   - Update pip: `pip install --upgrade pip`
   - Install system dependencies
   - Check Python version compatibility

### Log Files
- `startup.log` - System startup logs
- `advanced_vehicle_monitor.log` - Application logs
- Check console output for real-time information

## 📈 Performance Optimization

### Hardware Recommendations
- **CPU**: Multi-core processor (Intel i5+ or AMD Ryzen 5+)
- **RAM**: 8GB+ for multiple cameras
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: SSD for database operations

### Software Optimizations
- Use appropriate camera resolution
- Adjust confidence thresholds
- Enable batch processing
- Configure frame skipping for high FPS cameras

## 🔒 Security Considerations

- Change default ports if needed
- Use HTTPS in production
- Implement authentication
- Secure RTSP streams
- Regular security updates

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- Flask and Flask-SocketIO developers
- Contributors and testers

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check documentation
- Review troubleshooting guide

---

**Made with ❤️ for intelligent traffic monitoring**
