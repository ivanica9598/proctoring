import numpy as np
import cv2
from math import sqrt
from face_detector import FaceDetector
from face_aligner import FaceAligner


class FaceRecognizer:

    def __init__(self):
        self.net = cv2.dnn.readNetFromTorch("models/nn4.small2.v1.t7")
        self.face_detector = FaceDetector()
        self.face_aligner = FaceAligner()
        self.initial_image_encodings = None
        self.input_image_encodings = None

    def set_image(self, image, initial):
        face_boxes, face_confidences = self.face_detector.find_face_boxes(image)
        if len(face_boxes) == 1:
            encodings = []
            if not initial:
                new_img = self.face_aligner.align(image, face_boxes[0])
                #cv2.imshow('new', new_img)
                face_boxes, face_confidences = self.face_detector.find_face_boxes(new_img)
                encodings = self.calculate_encodings(new_img, face_boxes[0])
                #print(encodings)
                self.input_image_encodings = encodings
            else:
                encodings = self.calculate_encodings(image, face_boxes[0])
                self.initial_image_encodings = encodings
            return len(encodings) != 0
        return False

    def calculate_encodings(self, image, face_box):
        encodings = []
        # (startX, startY, endX, endY) = face_box.astype("int")
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
        dist = sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(self.initial_image_encodings, self.input_image_encodings)))
        print(dist)
        if dist <= 0.7:
            return True
        else:
            return False

    def test_recognizer(self):
        image_path = "images/face.jpg"
        student_image = cv2.imread(image_path)
        if self.set_image(student_image, True):
            cap = cv2.VideoCapture(0)
            while True:
                success, img = cap.read()
                if self.set_image(img, False):
                    if self.compare_faces():
                        cv2.putText(img, 'valid', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, 'fake', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                else:
                    print('Input image face not detected')

                cv2.imshow('output', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print('Student image is not valid')


recognizer = FaceRecognizer()
recognizer.test_recognizer()
