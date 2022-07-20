import numpy as np
import cv2
from math import sqrt
import dlib


class FaceRecognizer:

    def __init__(self, face_detector, face_aligner):
        self.net = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        # self.net = cv2.dnn.readNetFromTorch("models/nn4.small2.v1.t7")
        self.face_detector = face_detector
        self.face_aligner = face_aligner
        self.initial_image_encodings = None
        self.input_image_encodings = None

    def set_image(self, image, face_box, marks, initial):
        encodings = []
        if not initial:
            new_img = self.face_aligner.align(image, marks)
            face_boxes, face_confidences = self.face_detector.find_face_boxes(new_img)
            # encodings = self.calculate_encodings(new_img, face_boxes[0])
            encodings = np.array(self.net.compute_face_descriptor(image, marks, 1))
            self.input_image_encodings = encodings
        else:
            # encodings = self.calculate_encodings(image, face_box)
            encodings = np.array(self.net.compute_face_descriptor(image, marks, 1))
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

    def test(self, face_detector, mark_detector):
        image_path = "images/face.jpg"
        student_image = cv2.imread(image_path)

        face_boxes, face_confidences = face_detector.find_face_boxes(student_image)
        if len(face_boxes) == 1:
            marks = mark_detector.detect_landmarks(student_image, face_boxes[0])
            if self.set_image(student_image, face_boxes[0], marks, True):
                cap = cv2.VideoCapture(0)
                while True:
                    success, img = cap.read()
                    face_boxes, face_confidences = face_detector.find_face_boxes(img)
                    marks = mark_detector.detect_landmarks(img, face_boxes[0])
                    if len(face_boxes) == 1:
                        if self.set_image(img, face_boxes[0], marks, False):
                            if self.compare_faces():
                                print('valid')
                                cv2.putText(img, 'valid', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            else:
                                print('fake')
                                cv2.putText(img, 'fake', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        else:
                            print('Input image face not detected')
                    else:
                        print('Input image face not detected')

                    cv2.imshow('output', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        else:
            print('Student image is not valid')


