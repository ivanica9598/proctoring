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
        # cv2.imshow('Mask', mask)
        # mask specifies elements of the output array to be changed
        eye = cv2.bitwise_not(black_frame, original_frame.copy(), mask=mask)
        # cv2.imshow('Eye', eye)

        # Cropping
        margin = 5
        min_x = np.min(self.landmark_points[:, 0]) - margin
        max_x = np.max(self.landmark_points[:, 0]) + margin
        min_y = np.min(self.landmark_points[:, 1]) - margin
        max_y = np.max(self.landmark_points[:, 1]) + margin
        self.frame = eye[min_y:max_y, min_x:max_x]
        # cv2.imshow('Frame', self.frame)

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
        # bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp
        new_frame = cv2.bilateralFilter(self.frame, 5, 15, 15)
        # it erodes away the boundaries of foreground object(Always try to keep foreground in white)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        # If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        # new_frame = cv2.erode(new_frame, None, iterations=2) додато
        # new_frame = cv2.dilate(new_frame, None, iterations=4) додато
        # new_frame = cv2.medianBlur(new_frame, 3) додато
        # cv2.imshow('New frame', new_frame)
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
        # The eyeballs are segmented out and we can utilize cv2.findContours for finding them.
        # In OpenCV’s cv2.findContours() method, the object to find should be in white and the background is black.
        # all we need to do is now find the two largest contours and those should be our eyeballs
        # Find the largest contours on both sides of the midpoint bynsorting it with cv2.contourArea.
        # We can utilize cv2.moments to find the centers of the eyeballs.
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:] # last two items in the array
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
        if self.pupils_detected():
            return self.pupil_x / (self.center[0] * 2)
        else:
            return None

    def pupils_detected(self):
        return self.pupil_x is not None and self.pupil_y is not None









