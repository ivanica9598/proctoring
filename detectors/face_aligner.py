import numpy as np
import cv2


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.43, 0.43)):
        self.desiredLeftEye = desiredLeftEye

    def align(self, image, h, w, leftEyePts, rightEyePts):
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= w
        scale = desiredDist / dist

        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        center = (int(eyesCenter[0]), int(eyesCenter[1]))

        M = cv2.getRotationMatrix2D(center, angle, scale)

        tX = w * 0.5
        tY = h * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return output

    def test(self, face_detector, landmark_detector):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            if success:
                (h, w) = img.shape[:2]
                face_boxes = face_detector.detect_faces(img, h, w)
                landmark_detector.detect_landmarks(img, face_boxes[0][0])

                new_img = self.align(img, h, w, landmark_detector.get_left_eye_landmarks(),
                                     landmark_detector.get_right_eye_landmarks())

                (h, w) = new_img.shape[:2]
                face_boxes = face_detector.detect_faces(new_img, h, w)
                landmark_detector.detect_landmarks(new_img, face_boxes[0][0])
                landmark_detector.draw_landmarks(new_img)

                cv2.imshow('Face aligner', new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

