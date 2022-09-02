import cv2
import dlib
from detectors.helpers import shape_to_np
import numpy as np


class LandmarksDetector:
    def __init__(self):
        self.predictor = dlib.shape_predictor('detectors/landmarks_detector/shape_predictor_68_face_landmarks.dat')
        self.landmarks = None
        self.landmarks_np = None

    def detect_landmarks(self, img, face_box):
        try:
            rect = dlib.rectangle(face_box[0], face_box[1], face_box[2], face_box[3])
            marks = self.predictor(img, rect)
            self.landmarks = marks
            self.landmarks_np = shape_to_np(self.landmarks)
            return True, self.landmarks, self.landmarks_np
        except:
            return False, None, None

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
        if self.landmarks_np is not None:
            ids = [49, 50, 51, 52, 53, 61, 62, 63, 48, 54, 60, 64]
            landmarks = []
            for i in ids:
                landmarks.append(self.landmarks_np[i])
            return landmarks

    def get_bottom_lip_landmarks(self):
        if self.landmarks_np is not None:
            ids = [59, 58, 57, 56, 55, 67, 66, 65]
            landmarks = []
            for i in ids:
                landmarks.append(self.landmarks_np[i])
            return landmarks

    def get_head_pose_landmarks(self):
        if self.landmarks_np is not None:
            array = np.zeros((6, 2), dtype="double")
            # nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
            ids = [30, 8, 36, 45, 48, 54]
            for i in range(len(ids)):
                array[i] = self.landmarks_np[ids[i]]

            return array

    def draw_landmarks(self, image):
        if self.landmarks_np is not None:
            for mark in self.landmarks_np:
                cv2.circle(image, (mark[0], mark[1]), 2, (0, 255, 0), -1, cv2.LINE_AA)
