import numpy as np
import cv2


class Eye:
    def __init__(self, original_frame, landmarks):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil_x = None
        self.pupil_y = None
        self.iris_frame = None
        self.threshold = None
        self.landmark_points = landmarks

        self.analyze(original_frame)

    def analyze(self, original_frame):
        self.isolate_eye(original_frame)
        self.threshold = self.find_threshold()
        self.detect_iris()

    def isolate_eye(self, original_frame):
        height, width = original_frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [self.landmark_points], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, original_frame.copy(), mask=mask)

        # Cropping
        margin = 5
        min_x = np.min(self.landmark_points[:, 0]) - margin
        max_x = np.max(self.landmark_points[:, 0]) + margin
        min_y = np.min(self.landmark_points[:, 1]) - margin
        max_y = np.max(self.landmark_points[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def find_threshold(self):
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = self.image_processing(threshold)
            trials[threshold] = self.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def image_processing(self, threshold):
        # Returns a frame with a single element representing the iris
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(self.frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    @staticmethod
    def iris_size(frame):
        # Returns the percentage of space that the iris takes up on the surface of the eye.
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    def detect_iris(self):
        self.iris_frame = self.image_processing(self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.pupil_x = int(moments['m10'] / moments['m00'])
            self.pupil_y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass

    def get_pupil_coordinates(self):
        x = self.origin[0] + self.pupil_x
        y = self.origin[1] + self.pupil_y
        return x, y

    def get_horizontal_percentage(self):
        return self.pupil_x / (self.center[0] * 2)









