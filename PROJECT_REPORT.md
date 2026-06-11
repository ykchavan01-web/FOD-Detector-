# Runway Crack & Foreign Object Debris (FOD) Detector - Project Report

## Overview
The Runway Crack & FOD Detector is a computer vision system designed for airport runway inspection. It uses a smartphone camera as input, runs dual AI models (YOLOv8 and Edge Impulse) for object detection, and provides real-time alerts and visualization through a web-based dashboard.

## Architecture
The system follows a modular architecture:
```
[Phone Camera (IP Webcam)] 
        ↓
[Python Script (OpenCV) - video_capture.py]
        ↓
[AI Model Inference (YOLO + Edge Impulse) - ai_model.py]
        ↓
[Web Dashboard Alert (Flask) - app.py + dashboard/]
```

## Components

### 1. Main Application (`app.py`)
- Flask web server serving the dashboard
- Manages threading for video capture and detection
- Provides video stream endpoint (`/video_feed`)
- Provides detection results endpoint (`/detections`)
- Handles startup/shutdown and signal handling

### 2. Video Capture (`video_capture.py`)
- Handles phone camera streams via IP Webcam app or default webcam
- Uses OpenCV VideoCapture for frame acquisition
- Configurable resolution and FPS via config.py
- Thread-safe frame delivery via callback mechanism

### 3. AI Model (`ai_model.py`)
- Dual model inference: YOLOv8 (ultralytics) + Edge Impulse (TensorFlow/TFLite)
- YOLO: Object detection with bounding boxes
- Edge Impulse: Image classification (currently treated as full-frame detection)
- Non-Maximum Suppression to remove duplicate detections
- Detection visualization with bounding boxes and labels

### 4. Dashboard (`dashboard/`)
- **HTML** (`templates/index.html`): Bootstrap-based UI with video feed, detection list, alerts, and statistics
- **CSS** (`static/css/style.css`): Custom styling for components
- **JavaScript** (`static/js/script.js`): 
  - Periodic fetching of detection results via `/detections` endpoint
  - Real-time updates of detection list, alert status, and statistics
  - Video stream error handling and recovery
  - FPS calculation and display
  - Alert level determination based on detection types and confidence

### 5. Configuration (`config.py`)
- Centralized configuration for all system parameters
- Camera settings (URL, resolution, FPS)
- Detection thresholds (confidence, IoU)
- Model paths
- Alert thresholds
- Web server settings (host, port, debug)
- Target detection classes for runway inspection

## Features

### Core Functionality
- **Real-time video streaming** from phone camera via IP Webcam app
- **Dual AI model inference** for robust detection:
  - YOLOv8 nano for general object detection
  - Edge Impulse model for specialized crack/FOD detection
- **Web-based dashboard** with:
  - Live video feed with detection overlays
  - Real-time detection list with confidence scores
  - Alert status system (None/Low/Medium/High)
  - Statistics counters (total detections, cracks, FOD, FPS)
  - Responsive design (Bootstrap 5)

### Detection Capabilities
- Target classes: crack, fod (Foreign Object Debris), bolt, stone, debris, tool, trash
- Configurable confidence thresholds
- Model-specific visualization (green for YOLO, red for Edge Impulse)
- Detection history for temporal smoothing (configurable)

### System Features
- Thread-safe architecture for concurrent video capture and processing
- Graceful shutdown handling (SIGINT/SIGTERM)
- Error handling and recovery for video streams
- Automatic YOLOv8 model download on first run
- Support for custom-trained YOLO and Edge Impulse models

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```
Dependencies:
- opencv-python>=4.8.0
- ultralytics>=8.0.0
- flask>=3.0.0
- numpy>=2.0.0
- pillow>=10.0.0
- (Optional) tensorflow for Edge Impulse models

### Setup
1. Install IP Webcam app on smartphone (Android/iOS)
2. Start the IP webcam server and note the URL (typically `http://<phone-ip>:8080/video`)
3. Edit `config.py` to set `CAMERA_URL` to your phone's stream URL
4. (Optional) Place custom models:
   - YOLO: `models/yolov8n.pt` or custom path
   - Edge Impulse: `models/edge_impulse_model/` directory
5. Run the application: `python app.py`
6. Open web browser to `http://localhost:5000` for dashboard

## Model Details

### YOLOv8 Model
- Default: YOLOv8 nano (ultralytics)
- Automatically downloaded on first run if custom model not found
- Capable of detecting 80+ COCO classes, customized for runway inspection via class filtering
- Output: Bounding boxes, class labels, confidence scores

### Edge Impulse Model
- Expected formats: TensorFlow SavedModel, TensorFlow Lite (.tflite), or Keras H5
- Currently implemented as image classification (full-frame detection)
- For true object detection, would need to modify to use EI object detection exports
- Requires TensorFlow installation

## Data Flow
1. **Video Capture Thread**: Continuously grabs frames from camera, calls frame_callback
2. **Frame Callback**: Stores latest frame in thread-safe global variable
3. **Detection Thread**: 
   - Takes copy of latest frame
   - Runs YOLO detection → gets detections with bounding boxes
   - Runs Edge Impulse detection → gets classifications (treated as full-frame)
   - Applies NMS to remove duplicates
   - Updates global detections list
4. **Video Generation Thread**: 
   - Takes copy of latest frame and detections
   - Draws bounding boxes and labels on frame
   - Encodes as JPEG for HTTP streaming
5. **Web Server**: 
   - Serves dashboard HTML
   - Provides `/video_feed` endpoint for MJPEG stream
   - Provides `/detections` endpoint for JSON detection data
6. **Dashboard JavaScript**: 
   - Periodically fetches `/detections` (4x/sec)
   - Updates UI: detection list, alert status, statistics
   - Handles video stream errors and recovery

## Configuration Options

### Camera Settings
- `CAMERA_URL`: Phone IP webcam URL or `None` for default webcam
- Alternative format with credentials: `"http://user:pass@ip:port/video"`

### Video Settings
- `FRAME_WIDTH`: 640 (pixels)
- `FRAME_HEIGHT`: 480 (pixels)
- `TARGET_FPS`: 30

### Detection Settings
- `CONFIDENCE_THRESHOLD`: 0.5 (minimum confidence for detections)
- `IOU_THRESHOLD`: 0.45 (Non-Maximum Suppression threshold)

### Model Settings
- `YOLO_MODEL_PATH`: `"models/yolov8n.pt"` (custom model path)
- `EDGE_IMPULSE_MODEL_DIR`: `"models/edge_impulse_model"`

### Alert Settings
- `ALERT_THRESHOLDS`: 
  - low: 0.3
  - medium: 0.5
  - high: 0.7
- `TARGET_CLASSES`: ['crack', 'fod', 'bolt', 'stone', 'debris', 'tool', 'trash']

### Web Dashboard
- `FLASK_HOST`: '0.0.0.0' (listen on all interfaces)
- `FLASK_PORT`: 5000
- `FLASK_DEBUG`: True

### System
- `DETECTION_HISTORY_SIZE`: 30 (frames for temporal averaging)

## Project Structure
```
runway-fod-detector/
├── app.py                    # Main Flask application
├── video_capture.py          # Camera stream handling
├── ai_model.py               # AI model inference (YOLO + Edge Impulse)
├── config.py                 # Configuration parameters
├── requirements.txt          # Python dependencies
├── test_imports.py           # Import verification script
├── yolov8n.pt                # YOLOv8 nano model (auto-downloaded)
│
├── dashboard/
│   ├── templates/
│   │   └── index.html        # Main dashboard HTML
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css     # Dashboard styling
│   │   └── js/
│   │       └── script.js     # Dashboard interactivity
│   │
│   └── (models/ directory would contain custom models)
│
└── models/
    ├── yolov8n.pt            # Custom YOLO model (if exists)
    └── edge_impulse_model/   # Edge Impulse model export directory
```

## Strengths
1. **Modular Design**: Separation of concerns makes maintenance and extension easier
2. **Real-time Processing**: Threaded architecture maintains responsive UI
3. **Flexible Input**: Works with any IP webcam or USB webcam
4. **Dual Model Approach**: Combines general-purpose (YOLO) and specialized (Edge Impulse) detection
5. **User-Friendly Dashboard**: Clear visualization of detections, alerts, and statistics
6. **Error Handling**: Graceful degradation and recovery from common issues
7. **Configurable**: All parameters externalized to config.py

## Limitations & Improvement Opportunities
1. **Edge Impulse Integration**: Currently treats EI model as image classification rather than object detection
   - *Implement*: Use proper EI object detection exports for bounding box output
2. **Model Performance**: YOLOv8 nano may not be optimized for small crack/FOD detection
   - *Implement*: Train/customize YOLO model on runway-specific dataset
3. **Temporal Filtering**: Basic history size but no advanced tracking
   - *Implement*: Add object tracking (e.g., DeepSORT) for consistent detection IDs
4. **Alert System**: Basic threat level based on class and confidence
   - *Implement*: Incorporate location-specific rules (e.g., cracks on runway vs. grass)
5. **Scalability**: Single-instance design
   - *Implement*: Add support for multiple camera streams or distributed processing
6. **Recording/Logging**: No built-in video or detection logging
   - *Implement*: Add optional recording of video stream and detection events
7. **Unit Testing**: Limited test coverage
   - *Implement*: Add comprehensive unit and integration tests

## Usage Notes
- For best detection results, ensure good lighting and stable camera mounting
- The system is designed for runway inspection scenarios but can be adapted for other use cases
- Detection performance highly depends on model training quality and similarity to target objects
- Edge Impulse models require TensorFlow installation; install with `pip install tensorflow` if needed
- First run will download YOLOv8 nano model (~6MB) which may take a moment

## Conclusion
The Runway Crack & FOD Detector provides a solid foundation for computer-vision-based runway inspection. Its modular architecture, real-time processing capabilities, and user-friendly dashboard make it suitable for both immediate use and further customization. With targeted improvements to the Edge Impulse integration and model specialization, the system could achieve high accuracy for critical airport safety applications.