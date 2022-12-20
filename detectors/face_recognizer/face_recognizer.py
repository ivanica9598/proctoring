import numpy as np
import cv2
import dlib


class FaceRecognizer:

    def __init__(self):
        self.net = dlib.face_recognition_model_v1("detectors/face_recognizer/dlib_face_recognition_resnet_model_v1.dat")
        self.initial_image_encodings = None
        self.input_image_encodings = None
        self.counter = 0

        self.window = []
        self.window_counter = 0
        self.window_not_recognized = 0
        self.window_limit = 150
        self.cons = False

    def set_image(self, image, marks, initial):
        encodings = np.array(self.net.compute_face_descriptor(image, marks, 1))
        if initial:
            self.initial_image_encodings = encodings
        else:
            self.input_image_encodings = encodings
        return len(encodings) != 0

    def compare_faces(self, img, landmarks):
        valid = True
        if self.counter % 50 == 0:
            if self.set_image(img, landmarks, False):
                dist = np.linalg.norm(self.initial_image_encodings - self.input_image_encodings)
                if dist > 0.5:
                    valid = False
            else:
                valid = False
        self.counter = (self.counter + 1) % 50
        return valid

    @staticmethod
    def draw_result(frame, valid):
        if valid:
            cv2.putText(frame, 'Recognized', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Not recognized', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    def reset(self):
        problem = False

        if self.window_not_recognized >= 2:
            for frame in self.window:
                frame.msg += "Not recognized! "
                frame.valid = False
            problem = True

        self.window = []
        self.window_counter = 0
        self.window_not_recognized = 0
        self.cons = False
        self.counter = 0

        return problem

    def validate(self, input_frame, valid):
        problem = False
        self.window_counter = self.window_counter + 1
        self.window.append(input_frame)

        if not valid:
            self.window_not_recognized = self.window_not_recognized + 1
        if self.window_counter == self.window_limit:
            if self.window_not_recognized >= 2:
                for frame in self.window:
                    frame.msg += "Not recognized! "
                    frame.valid = False
                problem = True

            self.window_counter = 0
            self.window_not_recognized = 0
            self.window = []

        return problem
