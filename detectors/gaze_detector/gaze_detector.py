import cv2
from detectors.gaze_detector.eye import Eye


class GazeDetector:
    def __init__(self):
        self.frame = None
        self.left_eye = None
        self.right_eye = None
        self.horizontal_ratio = None
        self.vertical_ratio = None

        self.window = []
        self.window_counter = 0
        self.window_limit = 30
        self.window_looking_aside_counter = 0
        self.looking_cons_aside_buffer = []
        self.looking_cons_aside_counter = 0
        self.cons = False

    def draw_pupils(self, frame):
        color = (0, 255, 0)
        x_left, y_left = self.left_eye.get_pupil_coordinates()
        x_right, y_right = self.right_eye.get_pupil_coordinates()
        cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
        cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
        cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
        cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

    def check_frame(self, frame, left_eye_landmarks, right_eye_landmarks, closed):
        if closed:
            return False, "Eyes down"

        self.frame = frame
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.left_eye = Eye(frame, left_eye_landmarks)
        self.right_eye = Eye(frame, right_eye_landmarks)
        if self.left_eye.pupils_detected() and self.right_eye.pupils_detected():
            self.vertical_ratio = (self.left_eye.get_vertical_ratio() + self.right_eye.get_vertical_ratio()) / 2
            self.horizontal_ratio = (self.left_eye.get_horizontal_ratio() +
                                     self.right_eye.get_horizontal_ratio()) / 2
            # self.draw_pupils(self.frame)
            # cv2.imshow('Eyes', self.frame)
            if self.horizontal_ratio <= 0.35:
                return False, "Eyes right"
            elif self.horizontal_ratio >= 0.65:
                return False, "Eyes left"
            elif self.vertical_ratio <= 0.35:
                return False, "Eyes up"
            else:
                return True, "Eyes center"
        else:
            return True, None

    def reset(self):
        problem = False
        if self.cons:
            if self.looking_cons_aside_counter >= self.window_limit/2:
                for frame in self.looking_cons_aside_buffer:
                    frame.msg += "Looking aside! "
                    frame.valid = False
                problem = True
        elif self.window_counter >= 2 / 3 * self.window_limit and self.window_looking_aside_counter >= self.window_counter / 2:
            for i in range(self.window_counter):
                self.window[i].msg += "Looking aside! "
                self.window[i].valid = False
            problem = True

        self.window = []
        self.window_counter = 0
        self.window_looking_aside_counter = 0
        self.looking_cons_aside_buffer = []
        self.looking_cons_aside_counter = 0
        self.cons = False

        return problem

    def validate(self, input_frame, valid):
        problem = False

        if self.cons and not valid:
            self.looking_cons_aside_counter = self.looking_cons_aside_counter + 1
            self.looking_cons_aside_buffer.append(input_frame)
            return problem
        elif self.cons:
            self.cons = False
            if self.looking_cons_aside_counter >= self.window_limit/2:
                for frame in self.looking_cons_aside_buffer:
                    frame.msg += "Looking aside! "
                    frame.valid = False
                problem = True

        self.window_counter = self.window_counter + 1
        self.window.append(input_frame)

        if valid:
            self.looking_cons_aside_buffer = []
            self.looking_cons_aside_counter = 0
        else:
            self.looking_cons_aside_counter = self.looking_cons_aside_counter + 1
            self.looking_cons_aside_buffer.append(input_frame)
            self.window_looking_aside_counter = self.window_looking_aside_counter + 1

        if self.window_counter == self.window_limit:
            if self.looking_cons_aside_counter > 0:
                self.cons = True
                self.window_counter = self.window_counter - self.looking_cons_aside_counter
                self.window_looking_aside_counter = self.window_looking_aside_counter - self.looking_cons_aside_counter
            else:
                self.cons = False
            if self.window_looking_aside_counter >= self.window_counter / 3:
                for i in range(self.window_counter):
                    self.window[i].msg += "Looking aside! "
                    self.window[i].valid = False
                problem = True

            self.window_counter = 0
            self.window_looking_aside_counter = 0
            self.window = []

        return problem
