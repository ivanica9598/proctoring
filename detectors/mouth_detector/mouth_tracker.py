import cv2
import numpy as np


class MouthTracker:

    def __init__(self):
        self.dist_outer = [0]*5
        self.dist_inner = [0]*3
        self.initial_dist_outer = None
        self.initial_dist_inner = None
        self.input_dist_outer = None
        self.input_dist_inner = None

    def set_image(self, top_lip_landmarks, bottom_lip_landmarks, initial):
        for i in range(0, 5):
            self.dist_outer[i] = top_lip_landmarks[i][1] - bottom_lip_landmarks[i][1]
        for i in range(0, 3):
            self.dist_inner[i] = top_lip_landmarks[i+5][1] - bottom_lip_landmarks[i+5][1]

        x_dist_outer = top_lip_landmarks[8][0] - top_lip_landmarks[9][0]
        x_dist_inner = top_lip_landmarks[10][0] - top_lip_landmarks[11][0]

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
        if dist1 > 0.2 or dist2 > 0.1:
            return True

        return False

    def test(self, face_detector, landmarks_detector, student_image):
        (h, w) = student_image.shape[:2]
        image_face_boxes = face_detector.detect_faces(student_image, h, w)
        landmarks_detector.detect_landmarks(student_image, image_face_boxes[0][0])
        self.set_image(landmarks_detector.get_top_lip_landmarks(), landmarks_detector.get_bottom_lip_landmarks(), True)

        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()

            (h, w) = img.shape[:2]
            image_face_boxes = face_detector.detect_faces(img, h, w)
            landmarks_detector.detect_landmarks(img, image_face_boxes[0][0])
            self.set_image(landmarks_detector.get_top_lip_landmarks(), landmarks_detector.get_bottom_lip_landmarks(),
                           False)

            if self.compare_faces():
                cv2.putText(img, 'Talking', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Mouth tracker", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
