import numpy as np
import cv2


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
        self.head_cons_aside_buffer = []
        self.window_counter = 0
        self.window_head_aside_counter = 0
        self.head_cons_aside_counter = 0
        self.window_limit = 30
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
        elif (self.x > 165) or (0 > self.x < -176):
            result = True
        else:
            result = False

        # self.draw_result(img)
        return result

    def draw_result(self, img):
        if self.x is not None and self.y is not None and self.z is not None:
            cv2.putText(img, "x: " + str(np.round(self.x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "y: " + str(np.round(self.y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "z: " + str(np.round(self.z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def validate(self, img, valid, head_detector_buffer):
        problem = False

        if self.cons and not valid:
            self.head_cons_aside_counter = self.head_cons_aside_counter + 1
            self.head_cons_aside_buffer.append(img)
            return head_detector_buffer, problem
        elif self.cons:
            self.cons = False
            if self.head_cons_aside_counter >= 15:
                for i in range(self.head_cons_aside_counter):
                    cv2.putText(self.head_cons_aside_buffer[i], "Head aside", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    head_detector_buffer.append(self.head_cons_aside_buffer[i])
                problem = True

        self.window_counter = self.window_counter + 1
        self.window.append(img)

        if valid:
            self.head_cons_aside_buffer = []
            self.head_cons_aside_counter = 0
        else:
            self.head_cons_aside_counter = self.head_cons_aside_counter + 1
            self.head_cons_aside_buffer.append(img)
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
                    cv2.putText(self.window[i], "Head aside", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    head_detector_buffer.append(self.window[i])
                problem = True

            self.window_counter = 0
            self.window_head_aside_counter = 0
            self.window = []

        return head_detector_buffer, problem
