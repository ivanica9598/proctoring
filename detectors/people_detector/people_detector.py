import cv2
import numpy as np
from detectors.helpers import draw_box


class PeopleDetector:
    def __init__(self):
        self.neural_network = cv2.dnn.readNetFromCaffe('detectors/people_detector/MobileNetSSD_deploy.prototxt.txt',
                                                       'detectors/people_detector/MobileNetSSD_deploy.caffemodel')

        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.result = None

        self.window = []
        self.window_counter = 0
        self.window_limit = 30
        self.window_people_counter = 0
        self.people_cons_buffer = []
        self.people_cons_counter = 0
        self.cons = False

    def detect_people(self, frame, h, w):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.neural_network.setInput(blob)
        detections = self.neural_network.forward()

        self.result = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                class_name = self.classes[idx]
                if class_name == "person":
                    self.result.append((detections[0, 0, i, 3:7] * np.array([w, h, w, h]), confidence))
        valid = len(self.result) == 1
        # if not valid:
        #    self.draw_people(frame)
        return valid

    def draw_people(self, frame):
        if self.result is not None:
            for box in self.result:
                (left, top, right, bottom) = box[0].astype("int")
                draw_box(frame, [left, top, right, bottom], 'Person', box[1])

    def validate(self, img, valid, people_detector_buffer):
        if not valid:
            self.draw_people(img)

        problem = False

        if self.cons and not valid:
            self.people_cons_counter = self.people_cons_counter + 1
            self.people_cons_buffer.append(img)
            return people_detector_buffer, problem
        elif self.cons:
            self.cons = False
            if self.people_cons_counter >= 15:
                for frame in self.people_cons_buffer:
                    cv2.putText(frame, "Not 1 person!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    people_detector_buffer.append(frame)
                problem = True

        self.window_counter = self.window_counter + 1
        self.window.append(img)

        if valid:
            self.people_cons_buffer = []
            self.people_cons_counter = 0
        else:
            self.people_cons_counter = self.people_cons_counter + 1
            self.people_cons_buffer.append(img)
            self.window_people_counter = self.window_people_counter + 1

        if self.window_counter == self.window_limit:
            if self.people_cons_counter > 0:
                self.cons = True
                self.window_counter = self.window_counter - self.people_cons_counter
                self.window_people_counter = self.window_people_counter - self.people_cons_counter
            else:
                self.cons = False
            if self.window_people_counter >= self.window_counter / 3:
                for i in range(self.window_counter):
                    cv2.putText(self.window[i], "Not 1 person!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    people_detector_buffer.append(self.window[i])
                problem = True

            self.window_counter = 0
            self.window_people_counter = 0
            self.window = []

        return people_detector_buffer, problem

