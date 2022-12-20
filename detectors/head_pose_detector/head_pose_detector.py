import numpy as np
import cv2
import math


class HeadPoseDetector:
    def __init__(self):
        self.face_3d_model = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.initialized = False

        self.x = None
        self.y = None
        self.z = None

        self.window = []
        self.window_counter = 0
        self.window_limit = 30
        self.window_head_aside_counter = 0
        self.head_cons_aside_buffer = []
        self.head_cons_aside_counter = 0
        self.cons = False

    def initialize(self, width, height):
        focal_length = width
        center = (width / 2, height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4, 1))
        self.face_3d_model = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ], dtype=np.float64)
        self.initialized = True

    def detect_head_pose(self, img, h, w, image_points):
        if not self.initialized:
            self.initialize(w, h)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.face_3d_model, image_points,
                                                                      self.camera_matrix,
                                                                      self.dist_coeffs)
        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        self.x = angles[0]
        self.y = angles[1]
        self.z = angles[2]

        if self.y > 35 or self.y < -30:
            result = False
        elif self.x > 20 or self.x < -15:
            result = False
        else:
            result = True

        # self.draw_result(img)

        # for mark in image_points:
        #     cv2.circle(img, (int(mark[0]), int(mark[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)

        # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
        #                                             translation_vector, self.camera_matrix, self.dist_coeffs)

        # p1 = (int(image_points[0][0]), int(image_points[0][1]))
        # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # cv2.line(img, p1, p2, (0, 255, 0), 2)

        return result

    def draw_result(self, img):
        if self.x is not None and self.y is not None and self.z is not None:
            cv2.putText(img, "x: " + str(np.round(self.x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "y: " + str(np.round(self.y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "z: " + str(np.round(self.z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def reset(self):
        problem = False
        if self.cons:
            if self.head_cons_aside_counter >= self.window_limit/2:
                for frame in self.head_cons_aside_buffer:
                    frame.msg += "Head aside!"
                    frame.valid = False

                problem = True
        elif self.window_counter >= 2 / 3 * self.window_limit and self.window_head_aside_counter >= self.window_counter / 2:
            for i in range(self.window_counter):
                self.window[i].msg += "Head aside!"
                self.window[i].valid = False
            problem = True


        self.window = []
        self.window_counter = 0
        self.window_head_aside_counter = 0
        self.head_cons_aside_buffer = []
        self.head_cons_aside_counter = 0
        self.cons = False

        return problem

    def validate(self, input_frame, valid):
        problem = False

        if self.cons and not valid:
            self.head_cons_aside_counter = self.head_cons_aside_counter + 1
            self.head_cons_aside_buffer.append(input_frame)
            return problem
        elif self.cons:
            self.cons = False
            if self.head_cons_aside_counter >= self.window_limit/2:
                for frame in self.head_cons_aside_buffer:
                    frame.msg += "Head aside! "
                    frame.valid = False
                problem = True

        self.window_counter = self.window_counter + 1
        self.window.append(input_frame)

        if valid:
            self.head_cons_aside_buffer = []
            self.head_cons_aside_counter = 0
        else:
            self.head_cons_aside_counter = self.head_cons_aside_counter + 1
            self.head_cons_aside_buffer.append(input_frame)
            self.window_head_aside_counter = self.window_head_aside_counter + 1

        if self.window_counter == self.window_limit:
            if self.head_cons_aside_counter > 0:
                self.cons = True
                self.window_counter = self.window_counter - self.head_cons_aside_counter
                self.window_head_aside_counter = self.window_head_aside_counter - self.head_cons_aside_counter
            else:
                self.cons = False
            if self.window_head_aside_counter >= self.window_counter / 3:
                for i in range(self.window_counter):
                    self.window[i].msg += "Head aside! "
                    self.window[i].valid = False
                problem = True

            self.window_counter = 0
            self.window_head_aside_counter = 0
            self.window = []

        return problem

