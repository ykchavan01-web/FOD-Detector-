# Runway Crack & Foreign Object Debris (FOD) Detector

A computer vision system for detecting runway cracks and foreign object debris using phone camera input, YOLO and Edge Impulse AI models, with a web-based dashboard for alerts and visualization.

## Architecture

[Phone Camera (IP Webcam)] ---> [Python Script (OpenCV)] ---> [AI Model (YOLO + Edge Impulse)] ---> [Web Dashboard Alert]

## Features

- Real-time video stream from phone camera (via IP webcam app)
- Dual AI model inference (YOLOv8 for object detection + Edge Impulse model)
- Web-based dashboard showing live video feed with detection overlays
- Configurable detection thresholds and model parameters
- Alert visualization for detected cracks and FOD

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Install an IP webcam app on your phone (like IP Webcam for Android or similar for iOS)
2. Start the IP webcam server and note the URL (typically http://<phone-ip>:8080/video)
3. Update the CAMERA_URL in `video_capture.py` with your phone's stream URL
4. Run the application: `python app.py`
5. Open a web browser to http://localhost:5000 to view the dashboard

## Model Setup

### YOLO Model
The system uses YOLOv8 nano model by default, which will be automatically downloaded on first run.

### Edge Impulse Model
To use an Edge Impulse model:
1. Train and export your model from Edge Impulse
2. Place the exported model files in the `models/edge_impulse_model/` directory
3. Update the model loading path in `ai_model.py` if needed

## Project Structure

- `app.py` - Main Flask application
- `video_capture.py` - Handles phone camera stream via OpenCV
- `ai_model.py` - YOLO and Edge Impulse inference engine
- `dashboard/templates/index.html` - Main dashboard HTML
- `dashboard/static/css/style.css` - Dashboard styling
- `dashboard/static/js/script.js` - Dashboard interactivity
- `models/` - Directory for storing AI models
- `requirements.txt` - Python dependencies

## Configuration

Edit `config.py` (to be created) to adjust:
- Camera URL and settings
- Model confidence thresholds
- Detection classes (cracks, FOD types)
- Alert settings
- Dashboard refresh rates

## Notes

- For best results, ensure good lighting and stable camera mounting
- The system is designed for runway inspection scenarios
- Detection performance depends on model training quality and similarity to target objects