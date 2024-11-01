import numpy as np
import cv2


class SpeechDetector:

    def __init__(self):
        self.dist_outer = [0] * 5
        self.dist_inner = [0] * 3
        self.initial_dist_outer = None
        self.initial_dist_inner = None
        self.input_dist_outer = None
        self.input_dist_inner = None

        self.window = []
        self.window_counter = 0
        self.window_limit = 30
        self.window_open_counter = 0
        self.cons_open_buffer = []
        self.cons_open_counter = 0
        self.cons = False
        self.calculated = 0

    def initialize(self, student_image, student_top_lip, student_bottom_lip):
        self.set_image(student_image, student_top_lip, student_bottom_lip, True)

    def set_image(self, image, top_lip_landmarks, bottom_lip_landmarks, initial):
        for i in range(0, 5):
            self.dist_outer[i] = bottom_lip_landmarks[i][1] - top_lip_landmarks[i][1]
            # cv2.circle(image, (top_lip_landmarks[i][0], top_lip_landmarks[i][1]), 2, (0, 128, 255), -1, cv2.LINE_AA)
            # cv2.circle(image, (bottom_lip_landmarks[i][0], bottom_lip_landmarks[i][1]), 2, (0, 128, 255), -1, cv2.LINE_AA)
        for i in range(0, 3):
            self.dist_inner[i] = bottom_lip_landmarks[i + 5][1] - top_lip_landmarks[i + 5][1]
            # cv2.circle(image, (top_lip_landmarks[i + 5][0], top_lip_landmarks[i + 5][1]), 2, (0, 255, 0), -1, cv2.LINE_AA)
            # cv2.circle(image, (bottom_lip_landmarks[i + 5][0], bottom_lip_landmarks[i + 5][1]), 2, (0, 255, 0), -1, cv2.LINE_AA)
        x_dist_outer = top_lip_landmarks[9][0] - top_lip_landmarks[8][0]
        x_dist_inner = top_lip_landmarks[11][0] - top_lip_landmarks[10][0]

        # cv2.imshow("Lips", image)
        if initial:
            self.initial_dist_outer = self.dist_outer / x_dist_outer
            self.initial_dist_inner = self.dist_inner/ x_dist_inner
        else:
            self.input_dist_outer = self.dist_outer / x_dist_outer
            self.input_dist_inner = self.dist_inner/ x_dist_inner

    def initial_image_set(self):
        return self.initial_dist_outer is not None and self.initial_dist_inner is not None

    def input_image_set(self):
        return self.input_dist_outer is not None and self.input_dist_inner is not None

    def is_open(self, input_image, top_lip, bottom_lip):
        self.set_image(input_image, top_lip, bottom_lip, False)
        if self.initial_image_set() and self.input_image_set():
            dist1 = np.linalg.norm(self.initial_dist_outer - self.input_dist_outer)
            dist2 = np.linalg.norm(self.initial_dist_inner - self.input_dist_inner)
            if dist1 > 0.2 or dist2 > 0.2:
                return False
            else:
                return True

    def reset(self):
        problem = False

        if self.window_counter >= 2 / 3 * self.window_limit and self.window_open_counter > self.window_counter / 3:
            for i in range(self.window_counter):
                self.window[i].msg += "Speaking! "
                self.window[i].valid = False
                self.window[i].speech = True
            problem = True

        self.window = []
        self.window_counter = 0
        self.window_open_counter = 0
        self.cons_open_buffer = []
        self.cons_open_counter = 0
        self.cons = False
        self.calculated = 0

        return problem

    def validate(self, input_frame, valid):
        problem = False
        if self.cons and not valid:
            return problem

        self.window_counter = self.window_counter + 1
        self.window.append(input_frame)

        if valid:
            self.cons_open_counter = 0
            self.cons = False
            self.calculated = 0
        else:
            self.cons_open_counter = self.cons_open_counter + 1
            if self.cons_open_counter < 4:
                self.calculated = self.calculated + 1
                self.window_open_counter = self.window_open_counter + 1

        if self.window_counter == self.window_limit:
            if self.cons_open_counter > 0:
                self.cons = True
                self.window_counter = self.window_counter - self.cons_open_counter + self.calculated
                self.window_open_counter = self.window_open_counter - self.cons_open_counter + self.calculated
            else:
                self.cons = False
            if self.window_open_counter > self.window_counter / 4:
                for i in range(self.window_counter):
                    self.window[i].msg += "Speaking! "
                    self.window[i].valid = False
                    self.window[i].speech = True
                problem = True

            self.window = []
            self.window_counter = 0
            self.window_open_counter = 0
            self.calculated = 0

        return problem
