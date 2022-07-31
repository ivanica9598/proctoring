import cv2
import numpy as np


class PeopleCounter:
    def __init__(self):
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        self.neural_network = cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt.txt',
                                                       'models/MobileNetSSD_deploy.caffemodel')

    def test(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if success:
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                             (300, 300), 127.5)

                self.neural_network.setInput(blob)
                detections = self.neural_network.forward()

                for i in np.arange(0, detections.shape[2]):

                    # extract the confidence (i.e., probability) associated with the
                    # prediction
                    confidence = detections[0, 0, i, 2]
                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.3:
                        # extract the index of the class label from the `detections`,
                        # then compute the (x, y)-coordinates of the bounding box for
                        # the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # display the prediction
                        label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
                        print("[INFO] {}".format(label))
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      self.COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
