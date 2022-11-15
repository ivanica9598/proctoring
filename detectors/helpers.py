import numpy as np
import cv2


def shape_to_np(shape, dtype="int"):
    array = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        array[i] = (shape.part(i).x, shape.part(i).y)
    return array


def draw_box(frame, box, class_name, confidence):
    (startX, startY, endX, endY) = box[0], box[1], box[2], box[3]
    label = "{}: {:.2f}%".format(class_name, confidence * 100)
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

