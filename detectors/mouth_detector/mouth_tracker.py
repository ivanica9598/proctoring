import cv2
import numpy as np


class MouthTracker:

    def __init__(self):
        self.dist_outer = [0]*5
        self.dist_inner = [0]*3
        self.initial_dist_outer = None
        self.initial_dist_inner = None
        self.input_dist_outer = None
        self.input_dist_inner = None

    def initialize(self, student_top_lip, student_bottom_lip):
        self.set_image(student_top_lip, student_bottom_lip, True)

    def set_image(self, top_lip_landmarks, bottom_lip_landmarks, initial):
        for i in range(0, 5):
            self.dist_outer[i] = top_lip_landmarks[i][1] - bottom_lip_landmarks[i][1]
        for i in range(0, 3):
            self.dist_inner[i] = top_lip_landmarks[i+5][1] - bottom_lip_landmarks[i+5][1]

        x_dist_outer = top_lip_landmarks[8][0] - top_lip_landmarks[9][0]
        x_dist_inner = top_lip_landmarks[10][0] - top_lip_landmarks[11][0]

        if initial:
            self.initial_dist_outer = self.dist_outer / x_dist_outer
            self.initial_dist_inner = self.dist_inner / x_dist_inner
        else:
            self.input_dist_outer = self.dist_outer / x_dist_outer
            self.input_dist_inner = self.dist_inner / x_dist_inner

    def initial_image_set(self):
        return self.initial_dist_outer is not None and self.initial_dist_inner is not None

    def input_image_set(self):
        return self.input_dist_outer is not None and self.input_dist_inner is not None

    def compare_faces(self, top_lip, bottom_lip):
        self.set_image(top_lip, bottom_lip, False)
        if self.initial_image_set() and self.input_image_set():
            dist1 = np.linalg.norm(self.initial_dist_outer - self.input_dist_outer)
            dist2 = np.linalg.norm(self.initial_dist_inner - self.input_dist_inner)
            if dist1 > 0.2 or dist2 > 0.1:
                return False
            else:
                return True

