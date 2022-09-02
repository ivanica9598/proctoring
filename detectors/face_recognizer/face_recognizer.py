import numpy as np
import cv2
import dlib


class FaceRecognizer:

    def __init__(self):
        self.net = dlib.face_recognition_model_v1("detectors/face_recognizer/dlib_face_recognition_resnet_model_v1.dat")
        self.initial_image_encodings = None
        self.input_image_encodings = None
        self.counter = 0

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
        if self.counter % 100 == 0:
            if self.set_image(img, landmarks, False):
                dist = np.linalg.norm(self.initial_image_encodings - self.input_image_encodings)
                # print(dist)
                if dist > 0.7:
                    valid = False
            else:
                valid = False
        self.counter = (self.counter + 1) % 101
        return valid

    def draw_result(self, frame, valid):
        if valid:
            cv2.putText(frame, 'valid', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'fake', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


