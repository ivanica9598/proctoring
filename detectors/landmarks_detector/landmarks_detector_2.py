import cv2
import dlib
from detectors.helpers import shape_to_np


class LandmarksDetector2:
    def __init__(self):
        self.predictor = dlib.shape_predictor('detectors/landmarks_detector/shape_predictor_68_face_landmarks.dat')
        self.landmarks = None
        self.landmarks_np = None

    def detect_landmarks(self, img, face_box):
        rect = dlib.rectangle(face_box[0], face_box[1], face_box[2], face_box[3])
        marks = self.predictor(img, rect)
        self.landmarks = marks
        self.landmarks_np = shape_to_np(self.landmarks)
        return marks

    def get_landmarks_np(self):
        if self.landmarks_np is not None:
            return self.landmarks_np

    def get_left_eye_landmarks(self):
        if self.landmarks_np is not None:
            return self.landmarks_np[36:42]

    def get_right_eye_landmarks(self):
        if self.landmarks_np is not None:
            return self.landmarks_np[42:48]

    def get_top_lip_landmarks(self):
        ids = [49, 50, 51, 52, 53, 61, 62, 63, 48, 54, 60, 64]
        landmarks = []
        for i in ids:
            landmarks.append(self.landmarks_np[i])
        return landmarks

    def get_bottom_lip_landmarks(self):
        ids = [59, 58, 57, 56, 55, 67, 66, 65]
        landmarks = []
        for i in ids:
            landmarks.append(self.landmarks_np[i])
        return landmarks

    def draw_landmarks(self, image):
        for mark in self.landmarks_np:
            cv2.circle(image, (mark[0], mark[1]), 2, (0, 255, 0), -1, cv2.LINE_AA)

    def test(self, face_detector):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            (h, w) = img.shape[0:2]
            face_boxes = face_detector.detect_faces(img, h, w)
            self.detect_landmarks(img, face_boxes[0][0])
            self.draw_landmarks(img)

            cv2.imshow('Landmarks detector', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

