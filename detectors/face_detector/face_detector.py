from detectors.helpers import draw_box
import cv2
import dlib
from detectors.helpers import shape_to_np
import numpy as np


class FaceDetector:

    def __init__(self, desired_left_eye=(0.4, 0.4), desired_face_width=256, desiredFaceHeight=256):
        self.net = cv2.dnn.readNetFromTensorflow("detectors/face_detector/opencv_face_detector_uint8.pb",
                                                 "detectors/face_detector/opencv_face_detector.pbtxt")
        self.predictor = dlib.shape_predictor('detectors/face_detector/shape_predictor_68_face_landmarks.dat')
        self.landmarks = None
        self.landmarks_np = None
        self.result = None

        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desiredFaceHeight = desiredFaceHeight

        self.window = []
        self.window_counter = 0
        self.window_limit = 30
        self.face_cons_counter = 0
        self.face_cons_buffer = []
        self.window_face_counter = 0
        self.cons = False

    def detect_faces(self, image, h, w, threshold=0.6):
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        self.result = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                face_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (left, top, right, bottom) = face_box.astype("int")
                self.result.append(([left, top, right, bottom], confidence))

        # self.draw_faces(image)
        valid = len(self.result) == 1
        if valid:
            valid, self.landmarks, self.landmarks_np = self.detect_landmarks(image, self.result[0][0])

        return valid, self.landmarks, self.landmarks_np

    def align(self, image, left_eye_pts, right_eye_pts):
        desiredRightEyeX = 1.0 - self.desired_left_eye[0]

        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
        center = (int(eyes_center[0]), int(eyes_center[1]))

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desired_left_eye[0])
        desiredDist *= self.desired_face_width
        scale = desiredDist / dist
        M = cv2.getRotationMatrix2D(center, 0, scale)

        tX = self.desired_face_width * 0.5
        tY = self.desiredFaceHeight * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        (w, h) = (self.desired_face_width, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        parts = self.landmarks.parts()
        transformed_landmarks = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            rotated_point = M.dot(np.array((self.landmarks_np[i][0], self.landmarks_np[i][1], 1)))
            transformed_landmarks[i] = (int(rotated_point[0]), int(rotated_point[1]))
            parts[i] = dlib.point(transformed_landmarks[i][0], transformed_landmarks[i][1])
        self.landmarks_np = transformed_landmarks
        self.landmarks = dlib.full_object_detection(dlib.rectangle(0, 0, w, h), parts)

        # cv2.imshow('New', output)

        return output, self.landmarks, self.landmarks_np

    def draw_faces(self, image):
        if self.result is not None:
            for box in self.result:
                draw_box(image, box[0], 'Face', box[1])

    def detect_landmarks(self, img, face_box):
        try:
            rect = dlib.rectangle(face_box[0], face_box[1], face_box[2], face_box[3])
            marks = self.predictor(img, rect)
            self.landmarks = marks
            self.landmarks_np = shape_to_np(self.landmarks)
            # self.draw_landmarks(img)
            return True, self.landmarks, self.landmarks_np
        except:
            return False, None, None

    def get_landmarks_np(self):
        if self.landmarks_np is not None:
            return self.landmarks_np

    def get_left_eye_landmarks(self):
        if self.landmarks_np is not None:
            return self.landmarks_np[36:42]

    def get_right_eye_landmarks(self):
        if self.landmarks_np is not None:
            return self.landmarks_np[42:48]

    def get_top_lip_landmarks(self):
        if self.landmarks_np is not None:
            ids = [49, 50, 51, 52, 53, 61, 62, 63, 48, 54, 60, 64]
            landmarks = []
            for i in ids:
                landmarks.append(self.landmarks_np[i])
            return landmarks

    def get_bottom_lip_landmarks(self):
        if self.landmarks_np is not None:
            ids = [59, 58, 57, 56, 55, 67, 66, 65]
            landmarks = []
            for i in ids:
                landmarks.append(self.landmarks_np[i])
            return landmarks

    def get_head_pose_landmarks(self):
        if self.landmarks_np is not None:
            array = np.zeros((6, 2), dtype="double")
            # nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
            ids = [30, 8, 45, 36, 54, 48]
            for i in range(len(ids)):
                array[i] = self.landmarks_np[ids[i]]

            return array

    def draw_landmarks(self, image):
        if self.landmarks_np is not None:
            for mark in self.landmarks_np:
                cv2.circle(image, (mark[0], mark[1]), 2, (0, 255, 0), -1, cv2.LINE_AA)

    def reset(self):
        problem = False
        if self.cons:
            if self.face_cons_counter >= self.window_limit/2:
                for frame in self.face_cons_buffer:
                    frame.msg += "Not 1 person! "
                    frame.valid = False
                problem = True
        elif self.window_counter >= 2 / 3 * self.window_limit and self.window_face_counter >= self.window_counter / 2:
            for i in range(self.window_counter):
                self.window[i].msg += "Not 1 person! "
                self.window[i].valid = False
            problem = True

        self.window = []
        self.window_counter = 0
        self.window_face_counter = 0
        self.face_cons_buffer = []
        self.face_cons_counter = 0
        self.cons = False

        return problem

    def validate(self, input_frame, valid):
        if not valid:
            self.draw_faces(input_frame.img)

        problem = False

        if self.cons and not valid:
            self.face_cons_counter = self.face_cons_counter + 1
            self.face_cons_buffer.append(input_frame)
            return problem
        elif self.cons:
            self.cons = False
            if self.face_cons_counter >= self.window_limit/2:
                for frame in self.face_cons_buffer:
                    frame.msg += "Not 1 person! "
                    frame.valid = False
                problem = True

        self.window_counter = self.window_counter + 1
        self.window.append(input_frame)

        if valid:
            self.face_cons_buffer = []
            self.face_cons_counter = 0
        else:
            self.face_cons_counter = self.face_cons_counter + 1
            self.face_cons_buffer.append(input_frame)
            self.window_face_counter = self.window_face_counter + 1

        if self.window_counter == self.window_limit:
            if self.face_cons_counter > 0:
                self.cons = True
                self.window_counter = self.window_counter - self.face_cons_counter
                self.window_face_counter = self.window_face_counter - self.face_cons_counter
            else:
                self.cons = False
            if self.window_face_counter >= self.window_counter / 3:
                for i in range(self.window_counter):
                    self.window[i].msg += "Not 1 person! "
                    self.window[i].valid = False
                problem = True

            self.window_counter = 0
            self.window_face_counter = 0
            self.window = []

        return problem

