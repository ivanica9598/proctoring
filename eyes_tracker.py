import cv2
import numpy as np
from helpers import shape_to_np
import math
from PIL import ImageStat, Image


class EyesTracker:

    def __init__(self):
        self.result = None

        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def eye_on_mask(self, mask, side, shape):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        l = points[0][0]
        t = (points[1][1] + points[2][1]) // 2
        r = points[3][0]
        b = (points[4][1] + points[5][1]) // 2
        return mask, [l, t, r, b]

    def process_thresh(self, thresh):
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)
        return thresh

    def find_eyeball_position(self, end_points, cx, cy):
        x_ratio = (end_points[0] - cx) / (cx - end_points[2])
        y_ratio = (cy - end_points[1]) / (end_points[3] - cy)
        if x_ratio > 3:
            return 1
        elif x_ratio < 0.33:
            return 2
        elif y_ratio < 0.33:
            return 3
        else:
            return 0

    def contouring(self, thresh, mid, img, end_points, right=False):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
            pos = self.find_eyeball_position(end_points, cx, cy)
            return pos
        except:
            pass

    def print_eye_pos(self, img, left, right):
        if left == right and left != 0:
            text = ''
            if left == 1:
                print('Looking left')
                text = 'Looking left'
            elif left == 2:
                print('Looking right')
                text = 'Looking right'
            elif left == 3:
                print('Looking up')
                text = 'Looking up'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (30, 30), font,
                        1, (0, 255, 255), 2, cv2.LINE_AA)

    def test(self, face_detector, landmarks_detector):
        left = [36, 37, 38, 39, 40, 41]
        right = [42, 43, 44, 45, 46, 47]

        cap = cv2.VideoCapture(0)
        kernel = np.ones((9, 9), np.uint8)

        def nothing(x):
           pass

        cv2.namedWindow('image')
        cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

        while True:
            ret, img = cap.read()

            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_im)
            stat = ImageStat.Stat(pil_img)
            r, g, b = stat.mean
            brightness = math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
            # print(brightness)

            # frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # contrast = 1.25
            # br = 50
            # frame[:, :, 2] = np.clip(contrast * frame[:, :, 2] + br, 0, 255)
            # img = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

            input_image_face_boxes, _ = face_detector.find_face_boxes(img)
            rect = input_image_face_boxes[0]
            shape = landmarks_detector.detect_landmarks(img, rect)
            shape = shape_to_np(shape)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask, end_points_left = self.eye_on_mask(mask, left, shape)
            mask, end_points_right = self.eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]

            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            threshold = cv2.getTrackbarPos('threshold', 'image')
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = self.process_thresh(thresh)

            mid = int((shape[42][0] + shape[39][0]) // 2)
            eyeball_pos_left = self.contouring(thresh[:, 0:mid], mid, img, end_points_left)
            eyeball_pos_right = self.contouring(thresh[:, mid:], mid, img, end_points_right, True)
            self.print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)

            cv2.imshow('eyes', img)
            cv2.imshow("image", thresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


