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

    def detect_people(self, frame, h, w):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.neural_network.setInput(blob)
        detections = self.neural_network.forward()

        result = []
        counter = 0
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                class_name = self.classes[idx]
                if class_name == "person":
                    counter += 1
                    result.append((detections[0, 0, i, 3:7] * np.array([w, h, w, h]), confidence))

        self.result = result
        return counter == 1

    def draw_people(self, frame):
        if self.result is not None:
            for box in self.result:
                (left, top, right, bottom) = box[0].astype("int")
                draw_box(frame, [left, top, right, bottom], 'person', box[1])

    def test(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if success:
                (h, w) = frame.shape[:2]
                valid = self.detect_people(frame, h, w)
                self.draw_people(frame)
                if valid:
                    cv2.putText(frame, 'Valid', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Invalid', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("People detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
