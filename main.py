import cv2

from detectors.people_detector.people_detector import PeopleDetector
from detectors.face_detector.face_detector import FaceDetector
from detectors.landmarks_detector.landmarks_detector import LandmarksDetector
from detectors.head_pose_detector.head_pose_detector import HeadPoseDetector
from detectors.face_aligner import FaceAligner
from detectors.liveness_detector.liveness_detector import LivenessDetector
from detectors.eyes_detector.gaze_tracking import GazeTracker
from detectors.mouth_detector.mouth_tracker import MouthTracker
from detectors.face_recognizer.face_recognizer import FaceRecognizer


class ProctoringSystem:
    def __init__(self):
        self.people_detector = PeopleDetector()
        self.face_detector = FaceDetector()
        self.landmarks_detector = LandmarksDetector()
        self.head_pose_detector = HeadPoseDetector()
        self.face_aligner = FaceAligner()
        self.liveness_detector = LivenessDetector()
        self.gaze_tracker = GazeTracker()
        self.mouth_tracker = MouthTracker()
        self.face_recognizer = FaceRecognizer()

        self.student_image = None
        self.set_student_image()

    def set_student_image(self):
        image_path = "images/face.jpg"
        self.student_image = cv2.imread(image_path)

    def validate_student_image(self):
        (h, w) = self.student_image.shape[:2]
        if self.people_detector_validation(self.student_image, h, w):
            valid, face_box = self.face_detector_validation(self.student_image, h, w)
            if valid:
                self.face_detector.draw_faces(self.student_image)
                valid, landmarks, landmarks_np = self.landmarks_detector_validation(self.student_image, face_box)
                if valid:
                    top_lip = self.landmarks_detector.get_top_lip_landmarks()
                    bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                    self.mouth_tracker.initialize(top_lip, bottom_lip)
                    valid = self.face_recognizer.set_image(self.student_image, landmarks, True)
                    return valid
        return False

    def people_detector_validation(self, img, h, w):
        valid, counter = self.people_detector.detect_people(img, h, w)
        if valid:
            # self.people_detector.draw_people(img)
            return True
        else:
            self.people_detector.draw_people(img)
            self.report_problem(img, 'Detected ' + str(counter) + ' persons!')
            return False

    def face_detector_validation(self, img, h, w):
        valid, face_boxes = self.face_detector.detect_faces(img, h, w)
        if valid:
            # self.face_detector.draw_faces(img)
            return True, face_boxes[0][0]
        else:
            self.face_detector.draw_faces(img)
            self.report_problem(img, 'Detected more or less then one face!')
            return False, None

    def landmarks_detector_validation(self, img, face_box):
        valid, landmarks, landmarks_np = self.landmarks_detector.detect_landmarks(img, face_box)
        if valid:
            self.landmarks_detector.draw_landmarks(img)
            return valid, landmarks, landmarks_np
        else:
            self.report_problem(img, 'Can not detect facial landmarks!')
            return False, None, None

    def head_pose_detector_validation(self, img, h, w, image_points):
        valid, x, y, z = self.head_pose_detector.detect_head_pose(h, w, image_points)
        if not valid:
            self.report_problem(img, 'Head not forward!')

        # self.head_pose_detector.draw_result(img, x, y, z)
        return valid

    def face_aligner_validation(self, img, h, w, left_eye, right_eye):
        new_img = self.face_aligner.align(img, h, w, left_eye, right_eye)
        # cv2.imshow('Face aligner', new_img)
        return new_img

    def liveness_detector_validation(self, frame, left_eye, right_eye):
        closed, ear = self.liveness_detector.is_blinking(left_eye, right_eye)
        # self.liveness_detector.draw_result(frame, ear)
        if not closed:
            return True
        return False

    def gaze_detector_validation(self, img, frame, left_eye, right_eye):
        valid, msg = self.gaze_tracker.check_frame(frame, left_eye, right_eye)
        # self.gaze_tracker.draw_pupils(frame, msg)
        if not valid:
            self.report_problem(img, msg)
        return valid

    def mouth_tracker_validation(self, frame, top_lip, bottom_lip):
        valid = self.mouth_tracker.compare_faces(top_lip, bottom_lip)
        if not valid:
            self.report_problem(frame, 'Talking')
        return valid

    def face_recognizer_validation(self, frame, img, landmarks):
        valid = self.face_recognizer.compare_faces(img, landmarks)
        # self.face_recognizer.draw_result(frame, valid)
        if not valid:
            self.report_problem(frame, 'Not recognized')
        return valid

    def report_problem(self, img, msg):
        # cv2.imshow('Problem', img)
        cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    def start(self):
        valid = self.validate_student_image()
        if valid:
            cap = cv2.VideoCapture(0)
            while True:
                success, input_img = cap.read()
                if success:
                    (h, w) = input_img.shape[:2]
                    if self.people_detector_validation(input_img, h, w):
                        valid, face_box = self.face_detector_validation(input_img, h, w)
                        if valid:
                            self.face_detector.draw_faces(input_img)
                            valid, landmarks, landmarks_np = self.landmarks_detector_validation(input_img, face_box)
                            if valid:
                                image_points = self.landmarks_detector.get_head_pose_landmarks()
                                valid = self.head_pose_detector_validation(input_img, h, w, image_points)
                                if valid:
                                    left_eye = self.landmarks_detector.get_left_eye_landmarks()
                                    right_eye = self.landmarks_detector.get_right_eye_landmarks()
                                    new_img = self.face_aligner_validation(input_img, h, w, left_eye, right_eye)

                                    valid, face_box = self.face_detector_validation(new_img, h, w)
                                    if valid:
                                        valid, landmarks, landmarks_np = self.landmarks_detector_validation(new_img,
                                                                                                            face_box)
                                        if valid:
                                            left_eye = self.landmarks_detector.get_left_eye_landmarks()
                                            right_eye = self.landmarks_detector.get_right_eye_landmarks()
                                            valid = self.liveness_detector_validation(input_img, left_eye, right_eye)
                                            if valid:
                                                valid = self.gaze_detector_validation(input_img, new_img, left_eye, right_eye)
                                            if valid:
                                                top_lip = self.landmarks_detector.get_top_lip_landmarks()
                                                bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                                                valid = self.mouth_tracker_validation(input_img, top_lip, bottom_lip)
                                                if valid:
                                                    valid = self.face_recognizer_validation(input_img, new_img, landmarks)

                    cv2.imshow('output', input_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


proctoring_system = ProctoringSystem()
proctoring_system.start()
