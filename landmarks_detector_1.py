import cv2
import numpy as np
import tensorflow as tf
import dlib
from helpers import shape_to_np
from helpers import move_box
from helpers import get_square_box


class LandmarksDetector1:
    def __init__(self):
        self.model = tf.saved_model.load('models/pose_model')
        self.landmarks = None

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
        return obj

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
