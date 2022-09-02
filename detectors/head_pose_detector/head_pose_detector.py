import numpy as np
import cv2


class HeadPoseDetector:
    def __init__(self):
        self.face_3d_model = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.initialized = False

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

    def detect_head_pose(self, h, w, image_points):
        if not self.initialized:
            self.initialize(w, h)
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.face_3d_model, image_points,
                                                                      self.camera_matrix,
                                                                      self.dist_coeffs)

        rmat, jac = cv2.Rodrigues(rotation_vector)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0]
        y = angles[1]
        z = angles[2]

        if y > 25 or y < -25:
            result = False
        elif (x > 165) or (0 > x < -176):
            result = True
        else:
            result = False

        return result, x, y, z

    def draw_result(self, img, x, y, z):
        cv2.putText(img, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
