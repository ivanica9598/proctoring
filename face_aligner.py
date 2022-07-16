import numpy as np
import cv2
from face_detector import FaceDetector
from mark_detector import MarkDetector


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.42, 0.42),
                 desiredFaceWidth=256, desiredFaceHeight=None):

        self.detector = FaceDetector()
        self.predictor = MarkDetector()
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor.detect_marks(image, rect)

        leftEyePts = shape[36:42]
        rightEyePts = shape[42:48]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        center = (int(eyesCenter[0]), int(eyesCenter[1]))
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

    def test_aligner(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            face_boxes, face_confidences = self.detector.find_face_boxes(img)
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            new_img = self.align(img, face_boxes[0])

            cv2.imshow('input', img)
            cv2.imshow('output', new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# aligner = FaceAligner()
# aligner.test_aligner()
