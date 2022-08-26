import cv2
import numpy as np
import tensorflow as tf
import dlib
from detectors.helpers import shape_to_np
from detectors.helpers import move_box
from detectors.helpers import get_square_box


class LandmarksDetector1:
    def __init__(self):
        self.model = tf.saved_model.load('detectors/landmarks_detector/pose_model')
        self.landmarks = None
        self.landmarks_np = None

    def detect_landmarks(self, img, face):
        offset_y = int(abs((face[3] - face[1]) * 0.1))
        box_moved = move_box(face, [0, offset_y])
        facebox = get_square_box(box_moved)

        h, w = img.shape[:2]
        if facebox[0] < 0:
            facebox[0] = 0
        if facebox[1] < 0:
            facebox[1] = 0
        if facebox[2] > w:
            facebox[2] = w
        if facebox[3] > h:
            facebox[3] = h

        face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # # Actual detection.
        predictions = self.model.signatures["predict"](
            tf.constant([face_img], dtype=tf.uint8))

        # Convert predictions to landmarks.
        marks = np.array(predictions['output']).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))

        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        marks = marks.astype(np.uint)

        lst = dlib.points()
        for i in range(0, 68):
            lst.insert(i, dlib.point(marks[i][0], marks[i][1]))
        obj = dlib.full_object_detection(dlib.rectangle(face[0], face[1], face[2], face[3]), lst)

        self.landmarks = obj
        self.landmarks_np = shape_to_np(self.landmarks)
        return obj

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
            self.detect_landmarks(img, face_boxes[0])
            self.draw_landmarks(img)

            cv2.imshow('output', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
