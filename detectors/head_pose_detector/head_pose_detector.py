import numpy as np
import cv2


class HeadPoseDetector:
    def __init__(self):
        self.face_detector = None
        self.landmarks_detector = None

    def test(self, face_detector, landmarks_detector):
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
        height, width = img.shape[:2]

        face3Dmodel = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        while True:
            success, img = cap.read()
            (h, w) = img.shape[:2]
            face_boxes = face_detector.detect_faces(img, h, w)
            landmarks_detector.detect_landmarks(img, face_boxes[0][0])
            landmarks = landmarks_detector.get_landmarks_np()

            image_points = np.array([
                landmarks[30],  # Nose tip
                landmarks[8],  # Chin
                landmarks[36],  # Left eye left corner
                landmarks[45],  # Right eye right corne
                landmarks[48],  # Left Mouth corner
                landmarks[54]  # Right mouth corner
            ], dtype="double")

            for i in image_points:
                cv2.circle(img, (int(i[0]), int(i[1])), 4, (255, 0, 0), -1)

            (success, rotation_vector, translation_vector) = cv2.solvePnP(face3Dmodel, image_points, camera_matrix,
                                                                              dist_coeffs)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rotation_vector)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0]
            y = angles[1]
            z = angles[2]

            if y > 25 or y < -25:
                text = "Not forward"
            elif (x > 165) or (0 > x < -176):
                text = "Forward"
            else:
                text = "Not forward"

            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(img, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(1)
