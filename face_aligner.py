import numpy as np
import cv2


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.43, 0.43),
                 desiredFaceWidth=256, desiredFaceHeight=None):

        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, marks):
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
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        center = (int(eyesCenter[0]), int(eyesCenter[1]))

        M = cv2.getRotationMatrix2D(center, angle, scale)

        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
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

            cv2.imshow('Aligner input', img)
            cv2.imshow('Aligner output', new_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

