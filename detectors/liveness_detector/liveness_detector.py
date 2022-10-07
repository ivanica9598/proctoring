import cv2
import numpy as np

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3


class LivenessDetector:

    def __init__(self):
        self.result = None
        self.counter = 0
        self.total = 0

        self.window = []
        self.timer = -1
        self.last_val = None

    @staticmethod
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def is_blinking(self, frame, left_eye, right_eye):
        leftEAR = self.eye_aspect_ratio(left_eye)
        rightEAR = self.eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0
        blink = False
        if ear < EYE_AR_THRESH:
            self.counter += 1
            blink = True
        else:
            if self.counter >= EYE_AR_CONSEC_FRAMES:
                self.total += 1
            self.counter = 0

        # self.draw_result(frame, ear)
        return not blink

    def draw_result(self, frame, ear):
        cv2.putText(frame, "Blinks: {}".format(self.total), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Counter: {}".format(self.counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Ear: {}".format(ear), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def reset(self):
        self.total = 0
        self.window = []
        self.timer = -1
        self.last_val = None

    def validate(self, input_frame, liveness_detector_buffer, time_passed):
        problem = False

        timer = time_passed % 30
        if self.last_val != timer:
            self.last_val = timer
            self.timer = self.timer + 1

        self.window.append(input_frame)
        if self.timer == 29:
            self.timer = 0
            if self.total > 12 or self.total < 6:
                for frame in self.window:
                    frame.msg += "Not live face!"
                    liveness_detector_buffer.append(frame)
                problem = True

            self.total = 0
            self.window = []

        return liveness_detector_buffer, problem
