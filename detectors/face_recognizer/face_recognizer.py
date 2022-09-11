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
        self.window_not_recognized = False
        self.window_limit = 50
        self.cons = False

    def set_image(self, image, marks, initial):
        encodings = []
        if not initial:
            encodings = np.array(self.net.compute_face_descriptor(image, marks, 1))
            self.input_image_encodings = encodings
        else:
            encodings = np.array(self.net.compute_face_descriptor(image, marks, 1))
            self.initial_image_encodings = encodings
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

    def draw_result(self, frame, valid):
        if valid:
            cv2.putText(frame, 'Recognized', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Not recognized', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    def validate(self, img, valid, face_recognizer_buffer):
        problem = False

        self.window_counter = self.window_counter + 1
        self.window.append(img)

        if not valid:
            self.window_not_recognized = True
        if self.window_counter == self.window_limit:
            if self.window_not_recognized:
                for i in range(self.window_counter):
                    cv2.putText(self.window[i], "Not recognized!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
                    face_recognizer_buffer.append(self.window[i])
                problem = True

            self.window_counter = 0
            self.window_not_recognized = False
            self.window = []

        return face_recognizer_buffer, problem
