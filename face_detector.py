import cv2
import numpy as np


class FaceDetector:

    def __init__(self):
        # modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        # configFile = "models/deploy.prototxt"
        # net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.net = cv2.dnn.readNetFromTensorflow("models/opencv_face_detector_uint8.pb",
                                                 "models/opencv_face_detector.pbtxt")
        self.result_face_boxes = None
        self.result_confidences = None

    def find_face_boxes(self, image, threshold=0.5):
        (h, w) = image.shape[:2]
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        confidences = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                face_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = face_box.astype("int")
                faces.append([x, y, x1, y1])
                confidences.append(confidence)

        self.result_face_boxes = faces
        self.result_confidences = confidences
        return faces, confidences

    def draw_result(self, image):
        for face_box, confidence in zip(self.result_face_boxes, self.result_confidences):
            (startX, startY, endX, endY) = face_box[0], face_box[1], face_box[2], face_box[3]
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    def test(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            self.find_face_boxes(img)
            self.draw_result(img)
            cv2.imshow('Face detector', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


