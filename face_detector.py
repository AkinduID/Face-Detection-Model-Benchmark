# This codebase is built upon the initial work from the following repository:
# https://github.com/nodefluxio/face-detector-benchmark

# Modifications were made to adapt the code for the specific needs of this project,
# including the selection of evaluated models, evaluation metrics, and output formatting.

import numpy as np
import cv2
import mediapipe as mp
import os
from ultralytics import YOLO
import tensorflow as tf

class OpenCVHaarFaceDetector():
    def __init__(self, scaleFactor=1.3, minNeighbors=5, model_path='models/haarcascade_frontalface_default.xml'):
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
        self.face_cascade = cv2.CascadeClassifier(model_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.name="OpenCV Haar Cascade Face Detector"

    def detect_face(self, image):
        if image is None:
            raise ValueError("Input image is None")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors)
        faces_bbox = [[x, y, x + w, y + h] for x, y, w, h in faces]
        return np.array(faces_bbox)

class MediaPipeBlazeFaceDetector():
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1)
        self.name="MediaPipe BlazeFace Detector"

    def detect_face(self, image):
        if image is None:
            raise ValueError("Input image is None")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image)
        faces_bbox = []
        if results.detections:
            for detection in results.detections:
                box_x_min = int(detection.location_data.relative_bounding_box.xmin * image.shape[1])
                box_y_min = int(detection.location_data.relative_bounding_box.ymin * image.shape[0])
                box_width = int(detection.location_data.relative_bounding_box.width * image.shape[1])
                box_height = int(detection.location_data.relative_bounding_box.height * image.shape[0])
                faces_bbox.append([box_x_min, box_y_min, box_x_min + box_width, box_y_min + box_height])
        return np.array(faces_bbox)


class MediaPipeHolisticDetector():
    def __init__(self):
        self.face_detector = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.name = "MediaPipe Holistic Detector"
    def detect_face(self, image):
        if image is None:
            raise ValueError("Input image is None")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces_bbox = []
        results = self.face_detector.process(image)
        if results.face_landmarks:
            h, w, _ = image.shape
            face_landmarks = results.face_landmarks.landmark
            x_min = min([lm.x for lm in face_landmarks]) * w
            x_max = max([lm.x for lm in face_landmarks]) * w
            y_min = min([lm.y for lm in face_landmarks]) * h
            y_max = max([lm.y for lm in face_landmarks]) * h
            faces_bbox.append([int(x_min), int(y_min), int(x_max), int(y_max)])
        return np.array(faces_bbox)

class TensorFlowMobilNetSSDFaceDetector():
    def __init__(self,
                 det_threshold=0.3,
                 model_path='models/ssd/frozen_inference_graph_face.pb'):
        self.det_threshold = det_threshold
        self.detection_graph = tf.Graph()
        self.name = "MobileNet SSD Face Detector"

        # Load the frozen graph
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Create a session to run the graph
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    def detect_face(self, image):
    # Convert the image to RGB
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]  # Shape: (1, height, width, 3)

        # Get input and output tensors
        tensor_input = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Run the session to get the detection results
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={tensor_input: input_tensor.numpy()})  # Use the correctly shaped input tensor

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        filtered_score_index = np.argwhere(scores >= self.det_threshold).flatten()
        selected_boxes = boxes[filtered_score_index]

        faces = np.array([[int(y1 * image.shape[0]), int(x1 * image.shape[1]), int(y2 * image.shape[0]), int(x2 * image.shape[1])]
                        for y1, x1, y2, x2 in selected_boxes])

        return faces if len(faces) > 0 else np.array([])  # Return empty array if no faces found
    # Return empty array if no faces found


class YOLOFaceDetector():
    def __init__(self, model_path='models/yolov8n.pt', confidence_threshold=0.4):
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.name = "YOLO Face Detector"

    def detect_face(self, image):
        if image is None:
            raise ValueError("Input image is None")
        
        results = self.model.track(image, stream=True)
        faces_bbox = []

        for result in results:
            # Iterate over each box
            for box in result.boxes:
                # Check if confidence is greater than the threshold
                if box.conf[0] > self.confidence_threshold:
                    # Get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]
                    faces_bbox.append([int(x1), int(y1), int(x2), int(y2)])

        return np.array(faces_bbox)