from __future__ import division
import cv2
from eye import Eye


class GazeTracking(object):
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None

    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil_x)
            int(self.eye_left.pupil_y)
            int(self.eye_right.pupil_x)
            int(self.eye_right.pupil_y)
            return True
        except Exception:
            return False

    def analyze(self, landmarks):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        try:
            self.eye_left = Eye(frame, landmarks, 0)
            self.eye_right = Eye(frame, landmarks, 1)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame, landmarks):
        self.frame = frame
        self.analyze(landmarks)

    def pupil_left_coords(self):
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil_x
            y = self.eye_left.origin[1] + self.eye_left.pupil_y
            return (x, y)

    def pupil_right_coords(self):
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil_x
            y = self.eye_right.origin[1] + self.eye_right.pupil_y
            return (x, y)

    def horizontal_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil_x / (self.eye_left.center[0] * 2)
            pupil_right = self.eye_right.pupil_x / (self.eye_right.center[0] * 2)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil_y / (self.eye_left.center[1] * 2)
            pupil_right = self.eye_right.pupil_y / (self.eye_right.center[1] * 2)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_up(self):
        if self.pupils_located:
            return self.vertical_ratio() <= 0.25

    def is_center(self):
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def draw_pupils(self):
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

    def check_frame(self, frame, landmarks):
        self.refresh(frame, landmarks)

        if self.is_right():
            return "Looking right"
        elif self.is_left():
            return "Looking left"
        elif self.is_center():
            return "Looking center"
        if self.is_up():
            return "Looking up"

    def test(self, face_detector, landmarks_detector):
        webcam = cv2.VideoCapture(0)

        while True:
            _, frame = webcam.read()
            faces, _ = face_detector.find_face_boxes(frame)
            landmarks = landmarks_detector.detect_landmarks(frame, faces[0])
            self.refresh(frame, landmarks)

            frame = self.draw_pupils()
            text = ""

            if self.is_right():
                text = "Looking right"
            elif self.is_left():
                text = "Looking left"
            elif self.is_center():
                text = "Looking center"
            if self.is_up():
                text = "Looking up"

            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            cv2.imshow("Demo", frame)

            if cv2.waitKey(1) == 27:
                break

        webcam.release()
        cv2.destroyAllWindows()
