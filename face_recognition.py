import numpy as np
import cv2
from face_detection import get_face_detection_net, detect_faces
from math import sqrt


def get_face_encoding_net():
    encoding_file = "models/nn4.small2.v1.t7"
    encoding_net = cv2.dnn.readNetFromTorch(encoding_file)
    return encoding_net


def calculate_encodings(image, detection_net, embedding_net):
    detections = detect_faces(image, detection_net)
    encodings = []
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        (h, w) = image.shape[:2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW >= 20 and fH >= 20:
                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedding_net.setInput(face_blob)
                vec = embedding_net.forward()
                encodings = vec.flatten()
            else:
                print('No face detected')
        else:
            print('No face detected')
    else:
        print('No face detected')
    return encodings


face_detection_net = get_face_detection_net()
face_encoding_net = get_face_encoding_net()

imagePath = "face.jpg"
student_image = cv2.imread(imagePath)
known_encodings = calculate_encodings(student_image, face_detection_net, face_encoding_net)

if len(known_encodings) == 0:
    print('Student image is not valid')
else:
    cap = cv2.VideoCapture(0)
    while True:
        success, input_image = cap.read()
        input_encodings = calculate_encodings(input_image, face_detection_net, face_encoding_net)

        if len(input_encodings) > 0:
            normalized_known_encodings = known_encodings / np.linalg.norm(known_encodings)
            normalized_input_encodings = input_encodings / np.linalg.norm(input_encodings)
            dist = sqrt(sum((e1-e2)**2 for e1, e2 in zip(normalized_known_encodings, normalized_input_encodings)))
            print(dist)
            if dist <= 0.6:
                print('Valid student')
        cv2.imshow('output', input_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
