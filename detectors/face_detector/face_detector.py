import cv2
import numpy as np
from detectors.helpers import draw_box


class FaceDetector:

    def __init__(self):
        self.net = cv2.dnn.readNetFromTensorflow("detectors/face_detector/opencv_face_detector_uint8.pb",
                                                 "detectors/face_detector/opencv_face_detector.pbtxt")
        self.result = None

        self.window = []
        self.window_counter = 0
        self.window_limit = 30
        self.face_cons_counter = 0
        self.face_cons_buffer = []
        self.window_face_counter = 0
        self.cons = False

    def detect_faces(self, image, h, w, threshold=0.5):
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        self.result = []
        # The detections.shape[2] is the number of detected objects
        for i in range(0, detections.shape[2]):
            # 1. Batch ID
            # 2. Class ID
            # 3. Confidence
            # 4 - 7. Left, top, right, bottom

            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                face_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (left, top, right, bottom) = face_box.astype("int")
                self.result.append(([left, top, right, bottom], confidence))

        return len(self.result) == 1, self.result

    def draw_faces(self, image):
        if self.result is not None:
            for box in self.result:
                draw_box(image, box[0], 'Face', box[1])

    def reset(self, face_detector_buffer):
        problem = False
        if self.cons:
            if self.face_cons_counter >= 15:
                for frame in self.face_cons_buffer:
                    cv2.putText(frame, "Not 1 face!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    face_detector_buffer.append(frame)
                problem = True
        elif self.window_counter >= 2/3*self.window_limit and self.window_face_counter >= self.window_counter / 2:
            for i in range(self.window_counter):
                cv2.putText(self.window[i], "Not 1 face!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                face_detector_buffer.append(self.window[i])
            problem = True

        self.window = []
        self.window_counter = 0
        self.window_face_counter = 0
        self.face_cons_buffer = []
        self.face_cons_counter = 0
        self.cons = False

        return face_detector_buffer, problem

    def validate(self, img, valid, face_detector_buffer):
        if not valid:
            self.draw_faces(img)

        problem = False

        if self.cons and not valid:
            self.face_cons_counter = self.face_cons_counter + 1
            self.face_cons_buffer.append(img)
            return face_detector_buffer, problem
        elif self.cons:
            self.cons = False
            if self.face_cons_counter >= 15:
                for frame in self.face_cons_buffer:
                    cv2.putText(frame, "Not 1 face!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    face_detector_buffer.append(frame)
                problem = True

        self.window_counter = self.window_counter + 1
        self.window.append(img)

        if valid:
            self.face_cons_buffer = []
            self.face_cons_counter = 0
        else:
            self.face_cons_counter = self.face_cons_counter + 1
            self.face_cons_buffer.append(img)
            self.window_face_counter = self.window_face_counter + 1

        if self.window_counter == self.window_limit:
            if self.face_cons_counter > 0:
                self.cons = True
                self.window_counter = self.window_counter - self.face_cons_counter
                self.window_face_counter = self.window_face_counter - self.face_cons_counter
            else:
                self.cons = False
            if self.window_face_counter >= self.window_counter / 3:
                for i in range(self.window_counter):
                    cv2.putText(self.window[i], "Not 1 face!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    face_detector_buffer.append(self.window[i])
                problem = True

            self.window_counter = 0
            self.window_face_counter = 0
            self.window = []

        return face_detector_buffer, problem
