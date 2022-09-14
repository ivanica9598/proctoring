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

        self.people_detector_buffer = []
        self.face_detector_buffer = []
        self.head_detector_buffer = []
        self.liveness_detector_buffer = []
        self.eyes_detector_buffer = []
        self.mouth_detector_buffer = []
        self.face_recognizer_buffer = []

    def add_students(self):
        self.database.add_user_to_database("12345", "Petar", "Petrovic", "petar@gmail.com", "images/face1.jpg")
        self.database.add_user_to_database("67890", "Miroslav", "Mikic", "miroslav@gmail.com", "images/face1.jpg")
        self.database.add_user_to_database("16704", "Ivana", "Milivojevic", "ivana@gmail.com", "images/face1.jpg")

    def set_student_image(self, student_id_number):
        user, self.student_image = self.database.load_user(student_id_number)
        print(user["first_name"] + " " + user["last_name"] + ", " + user["id_number"])
        (h, w) = self.student_image.shape[:2]
        if self.people_detector_validation(self.student_image, h, w):
            valid, face_box = self.face_detector_validation(self.student_image, h, w)
            if valid:
                valid, landmarks, landmarks_np = self.landmarks_detector.detect_landmarks(self.student_image, face_box)
                if valid:
                    top_lip = self.landmarks_detector.get_top_lip_landmarks()
                    bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                    self.mouth_detector.initialize(top_lip, bottom_lip)
                    valid = self.face_recognizer.set_image(self.student_image, landmarks, True)
                    return valid
        return False

    def people_detector_validation(self, img, h, w):
        valid = self.people_detector.detect_people(img, h, w)
        self.people_detector_buffer, problem = self.people_detector.validate(img.copy(), valid, self.people_detector_buffer)

        if problem:
            self.report_problem(img, 'Not 1 person!')

        return valid

    def face_detector_validation(self, img, h, w):
        valid, face_boxes = self.face_detector.detect_faces(img, h, w)
        self.face_detector_buffer, problem = self.face_detector.validate(img.copy(), valid, self.face_detector_buffer)

        if problem:
            self.report_problem(img, 'Not 1 face!')

        if valid:
            return True, face_boxes[0][0]
        else:
            return False, None

    def head_pose_detector_validation(self, img, h, w, image_points):
        valid = self.head_pose_detector.detect_head_pose(img, h, w, image_points)
        self.head_detector_buffer, problem = self.head_pose_detector.validate(img.copy(), valid, self.head_detector_buffer)

        if problem:
            self.report_problem(img, 'Head aside!')

        return valid

    def liveness_detector_validation(self, img, left_eye, right_eye, time_passed):
        valid = self.liveness_detector.is_blinking(img, left_eye, right_eye)
        self.liveness_detector_buffer, problem = self.liveness_detector.validate(img.copy(), self.liveness_detector_buffer, time_passed)

        if problem:
            self.report_problem(img, "Not live face!")

        return valid

    def eyes_detector_validation(self, img, frame, left_eye, right_eye):
        valid, msg = self.eyes_detector.check_frame(frame, left_eye, right_eye)
        self.eyes_detector_buffer, problem = self.eyes_detector.validate(img.copy(), valid, self.eyes_detector_buffer)

        if problem:
            self.report_problem(img, "Looked aside!")

        return valid

    def mouth_detector_validation(self, img, top_lip, bottom_lip):
        valid = self.mouth_detector.is_open(top_lip, bottom_lip)
        self.mouth_detector_buffer, problem = self.mouth_detector.validate(img.copy(), valid, self.mouth_detector_buffer)

        if problem:
            self.report_problem(img, "Don't speak!")
        return valid

    def face_recognizer_validation(self, frame, img, landmarks):
        valid = self.face_recognizer.compare_faces(img, landmarks)
        self.face_recognizer_buffer, problem = self.face_recognizer.validate(frame.copy(), valid, self.face_recognizer_buffer)

        if problem:
            self.report_problem(frame, "Not recognized!")

        return valid

    @staticmethod
    def report_problem(img, msg):
        print(msg)

    @staticmethod
    def add_to_video(buffer, file_name, size):
        if len(buffer) != 0:
            out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), 8, size)
            for frame in buffer:
                out.write(frame)
            out.release()

    def start(self):
        valid = self.set_student_image("16704")
        if valid:
            cap = cv2.VideoCapture(0)
            _, input_img = cap.read()
            (h, w) = input_img.shape[:2]
            size = (w, h)
            start = time.time()
            while True:
                success, input_img = cap.read()
                if success:
                    time_passed = round(time.time() - start, 2)
                    cv2.putText(input_img, "Time: " + str(time_passed), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    if self.people_detector_validation(input_img, h, w):
                        valid, face_box = self.face_detector_validation(input_img, h, w)
                        if valid:
                            valid, landmarks, landmarks_np = self.landmarks_detector.detect_landmarks(input_img, face_box)
                            if valid:
                                image_points = self.landmarks_detector.get_head_pose_landmarks()
                                valid = self.head_pose_detector_validation(input_img, h, w, image_points)
                                if valid:
                                    left_eye = self.landmarks_detector.get_left_eye_landmarks()
                                    right_eye = self.landmarks_detector.get_right_eye_landmarks()
                                    new_img = self.face_aligner.align(input_img, h, w, left_eye, right_eye)

                                    valid, face_boxes = self.face_detector.detect_faces(new_img, h, w)
                                    if valid:
                                        face_box = face_boxes[0][0]
                                        valid, landmarks, landmarks_np = self.landmarks_detector.detect_landmarks(
                                            new_img, face_box)
                                        if valid:
                                            left_eye = self.landmarks_detector.get_left_eye_landmarks()
                                            right_eye = self.landmarks_detector.get_right_eye_landmarks()
                                            valid = self.liveness_detector_validation(input_img, left_eye, right_eye, int(time_passed))
                                            if valid:
                                                self.eyes_detector_validation(input_img, new_img, left_eye, right_eye)
                                            top_lip = self.landmarks_detector.get_top_lip_landmarks()
                                            bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                                            valid = self.mouth_detector_validation(input_img, top_lip, bottom_lip)
                                            if valid:
                                                self.face_recognizer_validation(input_img, new_img, landmarks)
                                            else:
                                                self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)
                                        else:
                                            self.liveness_detector.reset()
                                            self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                                            self.mouth_detector_buffer, _ = self.mouth_detector.reset(self.mouth_detector_buffer)
                                            self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)
                                    else:
                                        self.liveness_detector.reset()
                                        self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                                        self.mouth_detector_buffer, _ = self.mouth_detector.reset(self.mouth_detector_buffer)
                                        self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)
                                else:
                                    self.liveness_detector.reset()
                                    self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                                    self.mouth_detector_buffer, _ = self.mouth_detector.reset(self.mouth_detector_buffer)
                                    self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)
                            else:
                                self.head_detector_buffer, _ = self.head_pose_detector.reset(self.head_detector_buffer)
                                self.liveness_detector.reset()
                                self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                                self.mouth_detector_buffer, _ = self.mouth_detector.reset(self.mouth_detector_buffer)
                                self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)
                        else:
                            self.head_detector_buffer, _ = self.head_pose_detector.reset(self.head_detector_buffer)
                            self.liveness_detector.reset()
                            self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                            self.mouth_detector_buffer, _ = self.mouth_detector.reset(self.mouth_detector_buffer)
                            self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)
                    else:
                        self.face_detector_buffer, _ = self.face_detector.reset(self.face_detector_buffer)
                        self.head_detector_buffer, _ = self.head_pose_detector.reset(self.head_detector_buffer)
                        self.liveness_detector.reset()
                        self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                        self.mouth_detector_buffer, _ = self.mouth_detector.reset(self.mouth_detector_buffer)
                        self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)

                    cv2.imshow('Test', input_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.add_to_video(self.people_detector_buffer, 'people.avi', size)
            self.add_to_video(self.face_detector_buffer, 'face.avi', size)
            self.add_to_video(self.head_detector_buffer, 'head.avi', size)
            self.add_to_video(self.eyes_detector_buffer, 'eyes.avi', size)
            self.add_to_video(self.liveness_detector_buffer, 'liveness.avi', size)
            self.add_to_video(self.mouth_detector_buffer, 'mouth.avi', size)
            self.add_to_video(self.face_recognizer_buffer, 'recognizer.avi', size)


proctoring_system = ProctoringSystem()
# proctoring_system.add_students()
proctoring_system.start()
