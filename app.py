"""
Main Flask application for Runway Crack & FOD Detector
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import signal
import sys
from video_capture import VideoCapture
from ai_model import AIModel
import config

app = Flask(__name__, template_folder='dashboard/templates', static_folder='dashboard/static')

# Global variables for sharing data between threads
latest_frame = None
detections = []
frame_lock = threading.Lock()
detections_lock = threading.Lock()

# Thread control flags
running = False
video_thread = None
detection_thread = None

# Initialize components
video_capture = VideoCapture()
ai_model = AIModel()

# Set camera URL from config if provided
if config.CAMERA_URL is not None:
    video_capture.set_camera_url(config.CAMERA_URL)

def detect_objects():
    """Background thread for object detection"""
    global latest_frame, detections, running

    print("Detection thread started")
    while running:
        try:
            if latest_frame is not None:
                with frame_lock:
                    frame_copy = latest_frame.copy() if latest_frame is not None else None

                if frame_copy is not None:
                    # Run AI detection
                    detections_result = ai_model.detect(frame_copy)

                    with detections_lock:
                        detections.clear()
                        detections.extend(detections_result)

                    # Small delay to prevent overloading
                    time.sleep(0.1)
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.05)
        except Exception as e:
            print(f"Error in detection thread: {e}")
            time.sleep(0.5)  # Longer delay on error

def generate_frames():
    """Generate video frames for streaming"""
    global latest_frame, detections, running

    print("Video generation thread started")
    while running:
        try:
            if latest_frame is not None:
                with frame_lock:
                    frame = latest_frame.copy() if latest_frame is not None else None

                if frame is not None:
                    with detections_lock:
                        current_detections = detections.copy() if detections else []

                    # Draw detections on frame
                    annotated_frame = ai_model.draw_detections(frame, current_detections)

                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', annotated_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Error in frame generation: {e}")
            time.sleep(0.5)

def start_threads():
    """Start video capture and detection threads"""
    global running, video_thread, detection_thread

    if running:
        print("Threads already running")
        return

    running = True

    # Start video capture thread
    def frame_callback(frame):
        global latest_frame
        with frame_lock:
            latest_frame = frame

    video_thread = threading.Thread(target=video_capture.start_capture, args=(frame_callback,))
    video_thread.daemon = True
    video_thread.start()

    # Start detection thread
    detection_thread = threading.Thread(target=detect_objects)
    detection_thread.daemon = True
    detection_thread.start()

    print("All threads started")

def stop_threads():
    """Stop video capture and detection threads"""
    global running, video_thread, detection_thread

    if not running:
        print("Threads already stopped")
        return

    print("Stopping threads...")
    running = False

    # Stop video capture
    video_capture.stop_capture()

    # Wait for threads to finish (with timeout)
    if video_thread and video_thread.is_alive():
        video_thread.join(timeout=2.0)
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=2.0)

    print("Threads stopped")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print('\nReceived shutdown signal, cleaning up...')
    stop_threads()
    sys.exit(0)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Get latest detections as JSON"""
    with detections_lock:
        current_detections = detections.copy() if detections else []

    return jsonify({
        'detections': current_detections,
        'timestamp': time.time(),
        'count': len(current_detections)
    })

if __name__ == '__main__':
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start threads
        start_threads()

        # Run Flask app
        print(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT}")
        app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Error running application: {e}")
        stop_threads()
    finally:
        # Ensure cleanup on exit
        stop_threads()