"""
Configuration file for Runway Crack & FOD Detector
"""

# Camera Settings
# For phone IP webcam, use format: "http://<phone-ip>:<port>/video"
# Example: "http://192.168.1.100:8080/video"
CAMERA_URL = None  # Set to None to use default webcam (0)

# Alternative: If you need to specify username/password for some IP cameras
# CAMERA_URL = "http://username:password@192.168.1.100:8080/video"

# Video Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Model Settings
YOLO_MODEL_PATH = "models/yolov8n.pt"  # Custom model path (if exists)
EDGE_IMPULSE_MODEL_DIR = "models/edge_impulse_model"

# Alert Settings
ALERT_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.5,
    'high': 0.7
}

# Classes of interest for runway inspection
TARGET_CLASSES = ['crack', 'fod', 'bolt', 'stone', 'debris', 'tool', 'trash']

# Web Dashboard Settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Detection history size for averaging
DETECTION_HISTORY_SIZE = 30