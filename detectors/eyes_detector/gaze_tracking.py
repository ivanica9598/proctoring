from __future__ import division
import cv2
from detectors.eyes_detector.eye import Eye


class GazeTracker:
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.horizontal_ratio = None

    def set_new_frame(self, frame, left_eye_landmarks, right_eye_landmarks):
        self.frame = frame
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.eye_left = Eye(frame, left_eye_landmarks)
        self.eye_right = Eye(frame, right_eye_landmarks)

        if self.eye_left.pupils_detected() and self.eye_right.pupils_detected():
            self.horizontal_ratio = (self.eye_left.get_horizontal_percentage() +
                                     self.eye_right.get_horizontal_percentage()) / 2
        return self.horizontal_ratio is not None

    def draw_pupils(self, frame, text):
        cv2.putText(frame, str(text), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

        color = (0, 255, 0)
        x_left, y_left = self.eye_left.get_pupil_coordinates()
        x_right, y_right = self.eye_right.get_pupil_coordinates()
        cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
        cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
        cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
        cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        cv2.imshow("Gaze tracker", frame)

    def check_frame(self, frame, left_eye_landmarks, right_eye_landmarks):
        valid = self.set_new_frame(frame, left_eye_landmarks, right_eye_landmarks)
        if valid:
            if self.horizontal_ratio <= 0.35:
                return False, "Looking right"
            elif self.horizontal_ratio >= 0.65:
                return False, "Looking left"
            elif 0.35 < self.horizontal_ratio < 0.65:
                return True, "Looking center"
        else:
            return True, None
