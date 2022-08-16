import numpy as np
import cv2
from helpers import shape_to_np


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.43, 0.43)):
        self.desiredLeftEye = desiredLeftEye

    def align(self, image, marks):
        (img_h, img_w) = image.shape[:2]
        desiredFaceWidth = img_w
        desiredFaceHeight = img_h

        marks = shape_to_np(marks)
        leftEyePts = marks[36:42]
        rightEyePts = marks[42:48]

        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        center = (int(eyesCenter[0]), int(eyesCenter[1]))

        M = cv2.getRotationMatrix2D(center, angle, scale)

        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        return output

    def test(self, face_detector, landmark_detector):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()

            face_boxes, face_confidences = face_detector.find_face_boxes(img)
            landmarks = landmark_detector.detect_landmarks(img, face_boxes[0])
            new_img = self.align(img, landmarks)

            face_boxes, face_confidences = face_detector.find_face_boxes(new_img)
            landmarks = landmark_detector.detect_landmarks(new_img, face_boxes[0])
            landmark_detector.draw_landmarks(new_img)

            cv2.imshow('Aligner input', img)
            cv2.imshow('Aligner output', new_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

