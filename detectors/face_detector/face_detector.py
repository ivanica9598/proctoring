import cv2
import numpy as np
from detectors.helpers import draw_box


class FaceDetector:

    def __init__(self):
        # modelFile = "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
        # configFile = "models/face_detection/opencv_face_detector.prototxt"
        # self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        # Load the network and pass the model's layers and weights as its arguments
        self.net = cv2.dnn.readNetFromTensorflow("detectors/face_detector/opencv_face_detector_uint8.pb",
                                                 "detectors/face_detector/opencv_face_detector.pbtxt")
        self.result = None

    def detect_faces(self, image, h, w, threshold=0.5):
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        result = []
        # The detections.shape[2] is the number of detected objects
        for i in range(0, detections.shape[2]):
            # 1. Batch ID
            # 2. Class ID
            # 3. Confidence
            # 4 - 7. Left, top, right, bottom

            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                face_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (left, top, right, bottom) = face_box.astype("int")
                result.append(([left, top, right, bottom], confidence))

        self.result = result
        return self.result

    def draw_faces(self, image):
        if self.result is not None:
            for box in self.result:
                draw_box(image, box[0], 'face', box[1])

    def test(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if success:
                (h, w) = frame.shape[:2]
                self.detect_faces(frame, h, w)
                self.draw_faces(frame)

                cv2.imshow('Face detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


