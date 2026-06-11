"""
Test script to verify all modules can be imported correctly
"""

print("Testing imports...")

try:
    import cv2
    print(f"[PASS] OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"[FAIL] OpenCV import failed: {e}")

try:
    from ultralytics import YOLO
    print("[PASS] Ultralytics imported successfully")
except ImportError as e:
    print(f"[FAIL] Ultralytics import failed: {e}")

try:
    import flask
    print(f"[PASS] Flask version: {flask.__version__}")
except ImportError as e:
    print(f"[FAIL] Flask import failed: {e}")

try:
    import numpy
    print(f"[PASS] NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"[FAIL] NumPy import failed: {e}")

try:
    from video_capture import VideoCapture
    print("[PASS] VideoCapture imported successfully")
except ImportError as e:
    print(f"[FAIL] VideoCapture import failed: {e}")

try:
    from ai_model import AIModel
    print("[PASS] AIModel imported successfully")
except ImportError as e:
    print(f"[FAIL] AIModel import failed: {e}")

try:
    import config
    print("[PASS] Config imported successfully")
except ImportError as e:
    print(f"[FAIL] Config import failed: {e}")

print("\nInitializing AI Model...")
try:
    from ai_model import AIModel
    model = AIModel()
    print("[PASS] AI Model initialized successfully")
except Exception as e:
    print(f"[FAIL] AI Model initialization failed: {e}")

print("\nTest complete!")