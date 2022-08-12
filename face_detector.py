import cv2
import numpy as np


class FaceDetector:

    def __init__(self):
        # modelFile = "models/face_detection/res10_300x300_ssd_iter_140000.caffemodel"
        # configFile = "models/face_detection/opencv_face_detector.prototxt"
        # self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        # Load the network and pass the model's layers and weights as its arguments
        self.net = cv2.dnn.readNetFromTensorflow("models/face_detection/opencv_face_detector_uint8.pb",
                                                 "models/face_detection/opencv_face_detector.pbtxt")
        self.result_face_boxes = None
        self.result_confidences = None

    def find_face_boxes(self, image, threshold=0.5):
        # It returns a tuple of the number of rows, columns, and channels( if the image is color):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        confidences = []
        for i in range(0, detections.shape[2]):
            # 1. Batch ID
            # 2. Class ID
            # 3. Confidence
            # 4 - 7. Left, top, right, bottom

            # The detections.shape[2] is the number of detected objects
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                face_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (left, top, right, bottom) = face_box.astype("int")
                faces.append([left, top, right, bottom])
                confidences.append(confidence)

        self.result_face_boxes = faces
        self.result_confidences = confidences
        return faces, confidences

    def draw_result(self, image):
        for face_box, confidence in zip(self.result_face_boxes, self.result_confidences):
            self.draw_face(image, face_box, confidence)

    @staticmethod
    def draw_face(image, face_box, confidence):
        (left, top, right, bottom) = face_box[0], face_box[1], face_box[2], face_box[3]
        text = "{:.2f}%".format(confidence * 100)
        y = top - 10 if top - 10 > 10 else top + 10
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    def test(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            self.find_face_boxes(img)
            self.draw_result(img)
            cv2.imshow('Face detector', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


