import cv2
import time

from database.database import Database

from detectors.people_detector.people_detector import PeopleDetector
from detectors.face_detector.face_detector import FaceDetector
from detectors.landmarks_detector.landmarks_detector import LandmarksDetector
from detectors.head_pose_detector.head_pose_detector import HeadPoseDetector
from detectors.face_aligner import FaceAligner
from detectors.liveness_detector.liveness_detector import LivenessDetector
from detectors.eyes_detector.eyes_detector import EyesDetector
from detectors.mouth_detector.mouth_detector import MouthDetector
from detectors.face_recognizer.face_recognizer import FaceRecognizer


class ProctoringSystem:
    def __init__(self):
        self.database = Database()

        self.people_detector = PeopleDetector()
        self.face_detector = FaceDetector()
        self.landmarks_detector = LandmarksDetector()
        self.head_pose_detector = HeadPoseDetector()
        self.face_aligner = FaceAligner()
        self.liveness_detector = LivenessDetector()
        self.eyes_detector = EyesDetector()
        self.mouth_detector = MouthDetector()
        self.face_recognizer = FaceRecognizer()

        self.student_image = None

        self.eyes_detector_buffer = []
        self.head_detector_buffer = []

    def add_students(self):
        self.database.add_user_to_database("12345", "Petar", "Petrovic", "petar@gmail.com", "images/face.jpg")
        self.database.add_user_to_database("67890", "Miroslav", "Mikic", "miroslav@gmail.com", "images/face.jpg")
        self.database.add_user_to_database("16704", "Ivana", "Milivojevic", "ivana@gmail.com", "images/face.jpg")

    def set_student_image(self, student_id_number):
        user, self.student_image = self.database.load_user(student_id_number)
        print(user["first_name"] + " " + user["last_name"] + ", " + user["id_number"])
        (h, w) = self.student_image.shape[:2]
        if self.people_detector_validation(self.student_image, h, w):
            valid, face_box = self.face_detector_validation(self.student_image, h, w)
            if valid:
                self.face_detector.draw_faces(self.student_image)
                valid, landmarks, landmarks_np = self.landmarks_detector_validation(self.student_image, face_box)
                if valid:
                    top_lip = self.landmarks_detector.get_top_lip_landmarks()
                    bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                    self.mouth_detector.initialize(top_lip, bottom_lip)
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
        valid, face_boxes, counter = self.face_detector.detect_faces(img, h, w)
        if valid:
            # self.face_detector.draw_faces(img)
            return True, face_boxes[0][0]
        else:
            self.face_detector.draw_faces(img)
            self.report_problem(img, 'Detected ' + str(counter) + ' faces!')
            return False, None

    def landmarks_detector_validation(self, img, face_box):
        valid, landmarks, landmarks_np = self.landmarks_detector.detect_landmarks(img, face_box)
        if valid:
            # self.landmarks_detector.draw_landmarks(img)
            return valid, landmarks, landmarks_np
        else:
            self.report_problem(img, 'Can not detect facial landmarks!')
            return False, None, None

    def head_pose_detector_validation(self, img, h, w, image_points):
        valid = self.head_pose_detector.detect_head_pose(img, h, w, image_points)
        self.head_detector_buffer, problem = self.head_pose_detector.validate(img, valid, self.head_detector_buffer)

        if problem:
            self.report_problem(img, 'Look forward!')

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

    def eyes_detector_validation(self, img, frame, left_eye, right_eye):
        valid, msg = self.eyes_detector.check_frame(frame, left_eye, right_eye)
        self.eyes_detector_buffer, problem = self.eyes_detector.validate(img, valid, self.eyes_detector_buffer)

        if problem:
            self.report_problem(img, "Don't look aside!")

        return valid

    def mouth_detector_validation(self, frame, top_lip, bottom_lip):
        valid = self.mouth_detector.is_open(top_lip, bottom_lip)
        if not valid:
            self.report_problem(frame, 'Mouth open')
        return valid

    def face_recognizer_validation(self, frame, img, landmarks):
        valid = self.face_recognizer.compare_faces(img, landmarks)
        # self.face_recognizer.draw_result(frame, valid)
        if not valid:
            self.report_problem(frame, 'Not recognized')
        return valid

    def report_problem(self, img, msg):
        # cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(msg)

    def add_to_video(self, buffer, out):
        for frame in buffer:
            out.write(frame)

    def start(self):
        valid = self.set_student_image("16704")
        if valid:
            start = time.time()
            cap = cv2.VideoCapture(0)
            _, input_img = cap.read()
            (h, w) = input_img.shape[:2]
            size = (w, h)
            out1 = cv2.VideoWriter('eyes.avi', cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
            out2 = cv2.VideoWriter('head.avi', cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
            while True:
                success, input_img = cap.read()
                if success:
                    (h, w) = input_img.shape[:2]
                    time_passed = round(time.time() - start, 2)
                    cv2.putText(input_img, "Time: " + str(time_passed), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    if self.people_detector_validation(input_img, h, w):
                        valid, face_box = self.face_detector_validation(input_img, h, w)
                        if valid:
                            # self.face_detector.draw_faces(input_img)
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
                                                valid = self.eyes_detector_validation(input_img, new_img, left_eye,
                                                                                      right_eye)
                                            if valid:
                                                top_lip = self.landmarks_detector.get_top_lip_landmarks()
                                                bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                                                valid = self.mouth_detector_validation(input_img, top_lip, bottom_lip)
                                                if valid:
                                                    valid = self.face_recognizer_validation(input_img, new_img,
                                                                                            landmarks)
                    cv2.imshow('Test', input_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # self.add_to_video(self.eyes_detector_buffer, out1)
            self.add_to_video(self.head_detector_buffer, out2)
            out1.release()


proctoring_system = ProctoringSystem()
# proctoring_system.add_students()
proctoring_system.start()
