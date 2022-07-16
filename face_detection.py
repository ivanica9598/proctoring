import numpy as np
import cv2


def get_face_detection_net():
    # modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    # configFile = "models/deploy.prototxt"
    # net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    model_file = "models/opencv_face_detector_uint8.pb"
    config_file = "models/opencv_face_detector.pbtxt"
    model_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    return model_net


def detect_faces(img, net):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    return detections


def draw_face(img, box, confidence):
    (startX, startY, endX, endY) = box.astype("int")
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


def find_faces_on_image(image, detect_faces_net, min_confidence):
    (h, w) = image.shape[:2]
    faces = detect_faces(image, detect_faces_net)
    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > min_confidence:
            face_box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            draw_face(image, face_box, confidence)


def test_face_detection():
    detect_faces_net = get_face_detection_net()
    min_confidence = 0.5

    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        find_faces_on_image(img, detect_faces_net, min_confidence)
        cv2.imshow('output', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# test_face_detection()
