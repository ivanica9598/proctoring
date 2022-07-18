import cv2
import numpy as np
import tensorflow as tf


class LandmarksDetector1:
    def __init__(self):
        self.model = tf.saved_model.load('models/pose_model')

    def get_square_box(self, box):
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:
            return box
        elif diff > 0:
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

    def move_box(self, box, offset):
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    def detect_landmarks(self, img, face):
        offset_y = int(abs((face[3] - face[1]) * 0.1))
        box_moved = self.move_box(face, [0, offset_y])
        facebox = self.get_square_box(box_moved)

        h, w = img.shape[:2]
        if facebox[0] < 0:
            facebox[0] = 0
        if facebox[1] < 0:
            facebox[1] = 0
        if facebox[2] > w:
            facebox[2] = w
        if facebox[3] > h:
            facebox[3] = h

        face_img = img[facebox[1]: facebox[3],
                   facebox[0]: facebox[2]]
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

        return marks

    def draw_landmarks(self, image, landmarks):
        for mark in landmarks:
            cv2.circle(image, (mark[0], mark[1]), 2, (0, 255, 0), -1, cv2.LINE_AA)

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
