import cv2
from helpers import shape_to_np
import numpy as np


class MouthTracker:

    def __init__(self):
        self.outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
        self.inner_points = [[61, 67], [62, 66], [63, 65]]
        self.dist_outer = [0] * 5
        self.dist_inner = [0] * 3
        self.initial_dist_outer = None
        self.initial_dist_inner = None
        self.input_dist_outer = None
        self.input_dist_inner = None

    def set_image(self, landmarks, initial):
        landmarks = shape_to_np(landmarks)

        for i, (p1, p2) in enumerate(self.outer_points):
            self.dist_outer[i] = landmarks[p2][1] - landmarks[p1][1]
        for i, (p1, p2) in enumerate(self.inner_points):
            self.dist_inner[i] = landmarks[p2][1] - landmarks[p1][1]

        x_dist_outer = landmarks[54][0] - landmarks[48][0]
        x_dist_inner = landmarks[60][0] - landmarks[64][0]

        if initial:
            self.initial_dist_outer = self.dist_outer / x_dist_outer
            self.initial_dist_inner = self.dist_inner / x_dist_inner
        else:
            self.input_dist_outer = self.dist_outer / x_dist_outer
            self.input_dist_inner = self.dist_inner / x_dist_inner

    def compare_faces(self):
        dist1 = np.linalg.norm(self.initial_dist_outer - self.input_dist_outer)
        dist2 = np.linalg.norm(self.initial_dist_inner - self.input_dist_inner)

        print(dist1)
        print(dist2)
        if dist1 > 0.35 or dist2 > 0.3:
            return True

        return False

    def test(self, face_detector, landmarks_detector):
        image_path = "images/face.jpg"
        student_image = cv2.imread(image_path)

        image_face_boxes, _ = face_detector.find_face_boxes(student_image)
        landmarks = landmarks_detector.detect_landmarks(student_image, image_face_boxes[0])
        self.set_image(landmarks, True)

        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()

            image_face_boxes, _ = face_detector.find_face_boxes(img)
            landmarks = landmarks_detector.detect_landmarks(img, image_face_boxes[0])

            self.set_image(landmarks, False)
            if self.compare_faces():
                print('Mouth open')

            cv2.imshow("Output", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
