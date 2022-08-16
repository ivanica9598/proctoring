import cv2
import dlib
from helpers import shape_to_np


class LandmarksDetector2:
    def __init__(self):
        self.predictor = dlib.shape_predictor('models/facial_landmarks_detection/shape_predictor_68_face_landmarks.dat')
        self.landmarks = None

    def detect_landmarks(self, img, face_box):
        rect = dlib.rectangle(face_box[0], face_box[1], face_box[2], face_box[3])
        marks = self.predictor(img, rect)
        self.landmarks = marks
        return marks

    def draw_landmarks(self, image):
        landmarks = shape_to_np(self.landmarks)
        for mark in landmarks:
            cv2.circle(image, (mark[0], mark[1]), 2, (0, 255, 0), -1, cv2.LINE_AA)

    def test(self, face_detector):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            face_boxes, face_confidences = face_detector.find_face_boxes(img)
            self.detect_landmarks(img, face_boxes[0])
            self.draw_landmarks(img)

            cv2.imshow('output', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

