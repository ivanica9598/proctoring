import cv2
import numpy as np


class PeopleCounter:
    def __init__(self):
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        self.neural_network = cv2.dnn.readNetFromCaffe('models/people_detection/MobileNetSSD_deploy.prototxt.txt',
                                                       'models/people_detection/MobileNetSSD_deploy.caffemodel')

    def detect_persons(self, frame):
        (h, w) = frame.shape[:2]
        # (note: normalization is done via the authors of the MobileNet SSD implementation)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        self.neural_network.setInput(blob)
        detections = self.neural_network.forward()

        counter = 0
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                class_name = self.CLASSES[idx]
                if class_name == "person":
                    counter += 1

        return counter == 1

    @staticmethod
    def draw_box(frame, box, class_name, confidence):
        (startX, startY, endX, endY) = box.astype("int")
        label = "{}: {:.2f}%".format(class_name, confidence * 100)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def test(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if success:
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

                self.neural_network.setInput(blob)
                detections = self.neural_network.forward()
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.3:
                        idx = int(detections[0, 0, i, 1])
                        class_name = self.CLASSES[idx]
                        if class_name == "person":
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            self.draw_box(frame, box, class_name, confidence)

            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
