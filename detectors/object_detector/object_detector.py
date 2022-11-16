import cv2
import numpy as np
from detectors.helpers import draw_box


class ObjectDetector:
    def __init__(self):
        self.neural_network = cv2.dnn.readNetFromTensorflow('detectors/object_detector/frozen_inference_graph.pb',
                                                            'detectors/object_detector/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        with open('detectors/object_detector/COCO_labels.txt', 'r') as f:
            self.classes = f.read().split('\n')

        self.result = None
        self.window_limit = 30

        self.person_window = []
        self.person_window_counter = 0
        self.window_person_counter = 0
        self.person_cons_buffer = []
        self.person_cons_counter = 0
        self.person_cons = False

        self.cellphone_window = []
        self.cellphone_window_counter = 0
        self.window_cellphone_counter = 0
        self.cellphone_cons_buffer = []
        self.cellphone_cons_counter = 0
        self.cellphone_cons = False

    def detect(self, frame, h, w):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), size=(300, 300), mean=(104, 117, 123), swapRB=True)
        self.neural_network.setInput(blob)
        detections = self.neural_network.forward()

        person_cnt = 0
        cellphone_cnt = 0
        self.result = []

        for i in np.arange(0, detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score > 0.6:
                class_id = int(detections[0, 0, i, 1])
                class_name = self.classes[class_id]
                if class_name == "person" or class_name == "cellphone":
                    self.result.append((detections[0, 0, i, 3:7] * np.array([w, h, w, h]), score, class_name))
                    if class_name == "person":
                        person_cnt = person_cnt + 1
                    if class_name == "cellphone":
                        cellphone_cnt = cellphone_cnt + 1

        # self.draw(frame, "person")
        # self.draw(frame, "cellphone")

        return person_cnt == 1, cellphone_cnt == 0

    def draw(self, frame, class_name):
        if self.result is not None:
            for box in self.result:
                if box[2] == class_name:
                    (left, top, right, bottom) = box[0].astype("int")
                    draw_box(frame, [left, top, right, bottom], box[2], box[1])
                    # cv2.putText(frame, "Cellphone detected!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(frame, "Not 1 person!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def validate_person(self, input_frame, person_valid):
        if not person_valid:
            self.draw(input_frame.img, "person")

        problem = False

        if self.person_cons and not person_valid:
            self.person_cons_counter = self.person_cons_counter + 1
            self.person_cons_buffer.append(input_frame)
            return problem
        elif self.person_cons:
            self.person_cons = False
            if self.person_cons_counter >= 15:
                for frame in self.person_cons_buffer:
                    frame.msg += "Not 1 person! "
                    frame.valid = False
                problem = True

        self.person_window_counter = self.person_window_counter + 1
        self.person_window.append(input_frame)

        if person_valid:
            self.person_cons_buffer = []
            self.person_cons_counter = 0
        else:
            self.person_cons_counter = self.person_cons_counter + 1
            self.person_cons_buffer.append(input_frame)
            self.window_person_counter = self.window_person_counter + 1

        if self.person_window_counter == self.window_limit:
            if self.person_cons_counter > 0:
                self.person_cons = True
                self.person_window_counter = self.person_window_counter - self.person_cons_counter
                self.window_person_counter = self.window_person_counter - self.person_cons_counter
            else:
                self.person_cons = False

            if self.window_person_counter >= self.person_window_counter / 3:
                for i in range(self.person_window_counter):
                    self.person_window[i].msg += "Not 1 person! "
                    self.person_window[i].valid = False
                problem = True

            self.person_window_counter = 0
            self.window_person_counter = 0
            self.person_window = []

        return problem

    def validate_cellphone(self, input_frame, cellphone_valid):
        if not cellphone_valid:
            self.draw(input_frame.img, "cellphone")

        problem = False

        if self.cellphone_cons and not cellphone_valid:
            self.cellphone_cons_counter = self.cellphone_cons_counter + 1
            self.cellphone_cons_buffer.append(input_frame)
            return problem
        elif self.cellphone_cons:
            self.cellphone_cons = False
            if self.cellphone_cons_counter >= 15:
                for frame in self.cellphone_cons_buffer:
                    frame.msg += "Cellphone detected! "
                    frame.valid = False
                problem = True

        self.cellphone_window_counter = self.cellphone_window_counter + 1
        self.cellphone_window.append(input_frame)

        if cellphone_valid:
            self.cellphone_cons_buffer = []
            self.cellphone_cons_counter = 0
        else:
            self.cellphone_cons_counter = self.cellphone_cons_counter + 1
            self.cellphone_cons_buffer.append(input_frame)
            self.window_cellphone_counter = self.window_cellphone_counter + 1

        if self.cellphone_window_counter == self.window_limit:
            if self.cellphone_cons_counter > 0:
                self.cellphone_cons = True
                self.cellphone_window_counter = self.cellphone_window_counter - self.cellphone_cons_counter
                self.window_cellphone_counter = self.window_cellphone_counter - self.cellphone_cons_counter
            else:
                self.cellphone_cons = False

            if self.window_cellphone_counter >= self.cellphone_window_counter / 3:
                for i in range(self.cellphone_window_counter):
                    self.cellphone_window[i].msg += "Cellphone detected! "
                    self.cellphone_window[i].valid = False
                problem = True

            self.cellphone_window_counter = 0
            self.window_cellphone_counter = 0
            self.cellphone_window = []

        return problem

