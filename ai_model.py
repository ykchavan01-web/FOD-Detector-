"""
AI model module for runway crack and FOD detection
Handles both YOLO and Edge Impulse models
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import config

# TensorFlow import - make optional for environments without TF
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

class AIModel:
    def __init__(self):
        self.yolo_model = None
        self.edge_impulse_model = None
        self.class_names = []
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD

        # Initialize models
        self._load_yolo_model()
        self._load_edge_impulse_model()

        # Define class names for runway inspection
        # These can be customized based on your training data
        self.class_names = [
            'background',
            'crack',
            'fod',  # Foreign Object Debris
            'bolt',
            'stone',
            'debris',
            'tool',
            'trash'
        ]

    def _load_yolo_model(self):
        """Load YOLOv8 model"""
        try:
            # Try to load custom trained model first, fallback to default
            model_path = 'models/yolov8n.pt'
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                print(f"Loaded custom YOLO model from {model_path}")
            else:
                # Download and use YOLOv8 nano model
                self.yolo_model = YOLO('yolov8n.pt')
                print("Loaded YOLOv8 nano model (downloaded)")

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

    def _load_edge_impulse_model(self):
        """Load Edge Impulse exported model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - Edge Impulse model loading disabled")
            self.edge_impulse_model = None
            return

        try:
            # Edge Impulse can export in various formats
            # Check for common export formats
            model_dir = 'models/edge_impulse_model'

            # TensorFlow SavedModel format
            saved_model_path = os.path.join(model_dir, 'saved_model')
            if os.path.exists(saved_model_path):
                self.edge_impulse_model = tf.keras.models.load_model(saved_model_path)
                print(f"Loaded Edge Impulse model (SavedModel) from {saved_model_path}")
                return

            # TensorFlow Lite model
            tflite_path = os.path.join(model_dir, 'model.tflite')
            if os.path.exists(tflite_path):
                self.edge_impulse_model = tf.lite.Interpreter(model_path=tflite_path)
                self.edge_impulse_model.allocate_tensors()
                print(f"Loaded Edge Impulse model (TensorFlow Lite) from {tflite_path}")
                return

            # Keras H5 model
            h5_path = os.path.join(model_dir, 'model.h5')
            if os.path.exists(h5_path):
                self.edge_impulse_model = tf.keras.models.load_model(h5_path)
                print(f"Loaded Edge Impulse model (Keras H5) from {h5_path}")
                return

            print("No Edge Impulse model found in models/edge_impulse_model/")
            self.edge_impulse_model = None

        except Exception as e:
            print(f"Error loading Edge Impulse model: {e}")
            self.edge_impulse_model = None

    def detect(self, frame):
        """
        Run detection on frame using both YOLO and Edge Impulse models
        Returns list of detections: [{'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float, 'model': str}]
        """
        detections = []

        # Run YOLO detection
        if self.yolo_model is not None:
            yolo_dets = self._detect_yolo(frame)
            detections.extend(yolo_dets)

        # Run Edge Impulse detection
        if self.edge_impulse_model is not None:
            ei_dets = self._detect_edge_impulse(frame)
            detections.extend(ei_dets)

        # Apply non-maximum suppression to remove duplicates
        detections = self._apply_nms(detections, self.iou_threshold)

        return detections

    def _detect_yolo(self, frame):
        """Run YOLO detection on frame"""
        detections = []
        try:
            results = self.yolo_model(frame, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # Get class name
                        if class_id < len(self.yolo_model.names):
                            class_name = self.yolo_model.names[class_id]
                        else:
                            class_name = f"class_{class_id}"

                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class': class_name,
                            'confidence': float(confidence),
                            'model': 'YOLO'
                        })
        except Exception as e:
            print(f"Error in YOLO detection: {e}")

        return detections

    def _detect_edge_impulse(self, frame):
        """Run Edge Impulse detection on frame"""
        detections = []

        # Return empty detections if Edge Impulse model not available
        if self.edge_impulse_model is None:
            return detections

        try:
            # Preprocess frame for Edge Impulse model
            # This depends on how the model was trained - assuming image classification
            # For object detection, Edge Impulse exports different formats

            # Simple approach: resize and normalize
            input_size = (224, 224)  # Common size for EI image models
            resized = cv2.resize(frame, input_size)
            normalized = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(normalized, axis=0)

            if TENSORFLOW_AVAILABLE and isinstance(self.edge_impulse_model, tf.lite.Interpreter):
                # TensorFlow Lite model
                input_details = self.edge_impulse_model.get_input_details()
                output_details = self.edge_impulse_model.get_output_details()

                self.edge_impulse_model.set_tensor(input_details[0]['index'], input_data)
                self.edge_impulse_model.invoke()
                output_data = self.edge_impulse_model.get_tensor(output_details[0]['index'])

                # Process output (assuming classification output)
                predictions = output_data[0]
                top_class_id = np.argmax(predictions)
                confidence = float(predictions[top_class_id])

                if confidence > self.confidence_threshold:
                    # For simplicity, we'll treat this as a full-frame detection
                    # In reality, EI object detection would give bounding boxes
                    h, w = frame.shape[:2]
                    detections.append({
                        'bbox': [0, 0, w, h],  # Full frame - not ideal but functional
                        'class': self.class_names[top_class_id] if top_class_id < len(self.class_names) else f"class_{top_class_id}",
                        'confidence': confidence,
                        'model': 'Edge Impulse'
                    })
            elif TENSORFLOW_AVAILABLE:
                # Keras/TensorFlow model
                predictions = self.edge_impulse_model.predict(input_data, verbose=0)
                predictions = predictions[0]

                top_class_id = np.argmax(predictions)
                confidence = float(predictions[top_class_id])

                if confidence > self.confidence_threshold:
                    h, w = frame.shape[:2]
                    detections.append({
                        'bbox': [0, 0, w, h],  # Full frame
                        'class': self.class_names[top_class_id] if top_class_id < len(self.class_names) else f"class_{top_class_id}",
                        'confidence': confidence,
                        'model': 'Edge Impulse'
                    })

        except Exception as e:
            print(f"Error in Edge Impulse detection: {e}")

        return detections

    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) == 0:
            return detections

        # Convert to format expected by OpenCV NMSBoxes
        boxes = []
        scores = []
        class_ids = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # [x, y, width, height]
            scores.append(det['confidence'])
            # Simple class ID mapping - in practice, you'd want proper class mapping
            class_ids.append(hash(det['class']) % 100)  # Simple hash for class ID

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, iou_threshold)

        if len(indices) > 0:
            # Filter detections based on NMS results
            filtered_detections = [detections[i] for i in indices.flatten()]
            return filtered_detections
        else:
            return detections

    def draw_detections(self, frame, detections):
        """
        Draw detection bounding boxes and labels on frame
        Returns annotated frame
        """
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            confidence = det['confidence']
            model = det['model']

            # Choose color based on model
            if model == 'YOLO':
                color = (0, 255, 0)  # Green for YOLO
            else:
                color = (0, 0, 255)  # Red for Edge Impulse

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f'{class_name}: {confidence:.2f} ({model})'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated_frame