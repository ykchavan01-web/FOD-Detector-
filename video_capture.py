"""
Video capture module for handling phone camera streams
"""

import cv2
import threading
import time
import config

class VideoCapture:
    def __init__(self):
        self.camera_url = None
        self.cap = None
        self.running = False
        self.thread = None

    def set_camera_url(self, url):
        """Set the camera URL (for phone IP webcam)"""
        self.camera_url = url

    def start_capture(self, frame_callback):
        """
        Start capturing frames from camera
        Args:
            frame_callback: Function to call with each captured frame
        """
        self.running = True

        # If no camera URL set, try default webcam (0)
        if self.camera_url is None:
            self.camera_url = 0
            print(f"No camera URL specified, using default webcam: {self.camera_url}")

        self.cap = cv2.VideoCapture(self.camera_url)

        # Set video properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera/video stream: {self.camera_url}")
            self.running = False
            return

        print(f"Started video capture from: {self.camera_url}")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to grab frame")
                time.sleep(0.1)
                continue

            # Call the callback function with the frame
            if frame_callback:
                frame_callback(frame)

            # Control frame rate
            time.sleep(1.0 / config.TARGET_FPS)

        # Cleanup
        if self.cap:
            self.cap.release()
        print("Video capture stopped")

    def stop_capture(self):
        """Stop capturing frames"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)