from __future__ import division
import cv2
from detectors.eyes_detector.eye import Eye


class GazeTracker:
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.horizontal_ratio = None
        self.vertical_ratio = None

    def set_new_frame(self, frame, left_eye_landmarks, right_eye_landmarks):
        self.frame = frame
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.eye_left = Eye(frame, left_eye_landmarks)
        self.eye_right = Eye(frame, right_eye_landmarks)

        self.horizontal_ratio = (self.eye_left.get_horizontal_percentage() +
                                 self.eye_right.get_horizontal_percentage()) / 2

    def draw_pupils(self):
        frame = self.frame.copy()

        color = (0, 255, 0)
        x_left, y_left = self.eye_left.get_pupil_coordinates()
        x_right, y_right = self.eye_right.get_pupil_coordinates()
        cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
        cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
        cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
        cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

    def check_frame(self, frame, left_eye_landmarks, right_eye_landmarks):
        self.set_new_frame(frame, left_eye_landmarks, right_eye_landmarks)

        if self.horizontal_ratio <= 0.35:
            return "Looking right"
        elif self.horizontal_ratio >= 0.65:
            return "Looking left"
        elif 0.35 < self.horizontal_ratio < 0.65:
            return "Looking center"

    def test(self, face_detector, landmarks_detector):
        webcam = cv2.VideoCapture(0)

        while True:
            _, frame = webcam.read()
            (h, w) = frame.shape[:2]
            faces = face_detector.detect_faces(frame, h, w)
            landmarks_detector.detect_landmarks(frame, faces[0][0])

            text = self.check_frame(frame, landmarks_detector.get_left_eye_landmarks(),
                                    landmarks_detector.get_right_eye_landmarks())

            cv2.putText(frame, str(text), (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            frame = self.draw_pupils()

            cv2.imshow("Gaze tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()
