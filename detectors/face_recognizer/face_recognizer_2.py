import numpy as np
import cv2
from math import sqrt
import dlib


class FaceRecognizer2:

    def __init__(self):
        self.net = cv2.dnn.readNetFromTorch("detectors/face_recognizer/nn4.small2.v1.t7")
        self.initial_image_encodings = None
        self.input_image_encodings = None

    def set_image(self, image, face_box, initial):
        encodings = []
        if not initial:
            encodings = self.calculate_encodings(image, face_box)
            self.input_image_encodings = encodings
        else:
            encodings = self.calculate_encodings(image, face_box)
            self.initial_image_encodings = encodings
        return len(encodings) != 0

    def calculate_encodings(self, image, face_box):
        encodings = []
        face = image[face_box[1]:face_box[3], face_box[0]:face_box[2]]
        (fH, fW) = face.shape[:2]
        if fW >= 20 and fH >= 20:
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                              (96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(face_blob)
            vec = self.net.forward()
            encodings = vec.flatten()
            encodings = encodings / np.linalg.norm(encodings)
        return encodings

    def compare_faces(self):
        dist = np.linalg.norm(self.initial_image_encodings - self.input_image_encodings)
        # dist = sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(self.initial_image_encodings, self.input_image_encodings)))
        print(dist)
        if dist <= 0.7:
            return True
        else:
            return False

    def test(self, face_detector, mark_detector, face_aligner, student_image):
        (h, w) = student_image.shape[:2]

        face_boxes = face_detector.detect_faces(student_image, h, w)
        if len(face_boxes) == 1:
            marks = mark_detector.detect_landmarks(student_image, face_boxes[0][0])
            if self.set_image(student_image, face_boxes[0][0], True):
                cap = cv2.VideoCapture(0)
                counter = 0
                while True:
                    success, img = cap.read()
                    if counter % 2000 == 0:
                        counter = 0
                        (h, w) = img.shape[:2]
                        face_boxes = face_detector.detect_faces(img, h, w)
                        marks = mark_detector.detect_landmarks(img, face_boxes[0][0])

                        new_img = face_aligner.align(img, h, w,
                                                     mark_detector.get_left_eye_landmarks(),
                                                     mark_detector.get_right_eye_landmarks())
                        (h, w) = new_img.shape[:2]
                        face_boxes = face_detector.detect_faces(new_img, h, w)
                        if len(face_boxes) == 1:
                            if self.set_image(img, face_boxes[0][0], False):
                                if self.compare_faces():
                                    cv2.putText(img, 'valid', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                else:
                                    cv2.putText(img, 'fake', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            else:
                                print('Input image face not detected')
                        else:
                            print('Input image face not detected')
                    #else:
                     #   counter = counter + 1

                    cv2.imshow('Face recognizer', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        else:
            print('Student image is not valid')
