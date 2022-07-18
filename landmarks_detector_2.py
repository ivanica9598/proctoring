import cv2
import numpy as np
import dlib


class LandmarksDetector2:
    def __init__(self):
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def detect_landmarks(self, img, face_box):
        rect = dlib.rectangle(face_box[0], face_box[1], face_box[2], face_box[3])
        marks = self.predictor(img, rect)
        # marks = self.shape_to_np(marks)
        return marks

    def draw_landmarks(self, image, landmarks):
        for i in range(0, 68):
            cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1, cv2.LINE_AA)

    def test(self, face_detector):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            face_boxes, face_confidences = face_detector.find_face_boxes(img)
            marks = self.detect_landmarks(img, face_boxes[0])
            self.draw_landmarks(img, marks)

            cv2.imshow('output', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

