import cv2
import time
import math

from database.database import Database
from detectors.object_detector.object_detector import ObjectDetector
from detectors.face_detector.face_detector import FaceDetector
from detectors.head_pose_detector.head_pose_detector import HeadPoseDetector
from detectors.liveness_detector.liveness_detector import LivenessDetector
from detectors.eyes_tracker.eyes_tracker import EyesTracker
from detectors.speech_detector.speech_detector import SpeechDetector
from detectors.face_recognizer.face_recognizer import FaceRecognizer


class Frame:
    def __init__(self, img, frame_time=""):
        self.img = img
        self.msg = ""
        self.time = frame_time
        self.valid = True


class ProctoringSystem:
    def __init__(self):
        self.database = Database()

        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.head_pose_detector = HeadPoseDetector()
        self.liveness_detector = LivenessDetector()
        self.eyes_tracker = EyesTracker()
        self.speech_detector = SpeechDetector()
        self.face_recognizer = FaceRecognizer()

        self.student = None
        self.student_image = None
        self.test = None

        self.invalid_buffer = []
        self.main_buffer = []
        self.frame_counter = 0
        self.warning = ""

    def add_students(self):
        self.database.add_student("12345", "Petar", "Petrovic", "petar@gmail.com", "images/face1.jpg")
        self.database.add_student("67890", "Miroslav", "Mikic", "miroslav@gmail.com", "images/face1.jpg")
        self.database.add_student("16704", "Ivana", "Milivojevic", "ivana@gmail.com", "images/face1.jpg")

    def add_tests(self):
        self.database.add_test("Math-test1", 240)
        self.database.add_test("Math-test2", 120)
        self.database.add_test("Math-test3", 180)

    def load_student(self, student_id):
        self.student, self.student_image = self.database.load_student(student_id)
        print("Welcome " + self.student["first_name"] + " " + self.student["last_name"] + "!")

        (h, w) = self.student_image.shape[:2]
        frame = Frame(self.student_image)

        valid, _, _ = self.face_detector_validation(frame, h, w)
        if valid:
            left_eye = self.face_detector.get_left_eye_landmarks()
            right_eye = self.face_detector.get_right_eye_landmarks()
            self.student_image, landmarks, _ = self.face_detector.align(self.student_image, left_eye, right_eye)
            # cv2.imshow('Student', self.student_image)

            top_lip = self.face_detector.get_top_lip_landmarks()
            bottom_lip = self.face_detector.get_bottom_lip_landmarks()
            self.speech_detector.initialize(top_lip, bottom_lip)

            valid = self.face_recognizer.set_image(self.student_image, landmarks, True)

        return valid

    def load_test(self, test_id):
        self.test = self.database.load_test(test_id)

    def object_detector_validation(self, input_frame, h, w):
        person_valid, cellphone_valid = self.object_detector.detect(input_frame.img, h, w)
        person_problem = self.object_detector.validate_person(input_frame, person_valid)
        cellphone_problem = self.object_detector.validate_cellphone(input_frame, cellphone_valid)

        if person_problem:
            self.warning += 'Not 1 person!'
        if cellphone_problem:
            self.warning += 'Cellphone detected!'

        return person_valid

    def face_detector_validation(self, input_frame, h, w):
        valid, landmarks, landmarks_np = self.face_detector.detect_faces(input_frame.img, h, w)
        problem = self.face_detector.validate(input_frame, valid)

        if problem:
            self.warning += 'Not 1 face!'

        return valid, landmarks, landmarks_np

    def head_pose_detector_validation(self, input_frame, h, w, image_points):
        valid = self.head_pose_detector.detect_head_pose(input_frame.img, h, w, image_points)
        problem = self.head_pose_detector.validate(input_frame, valid)

        if problem:
            self.warning += 'Head aside!'

        return valid

    def liveness_detector_validation(self, input_frame, left_eye, right_eye, time_passed):
        valid = self.liveness_detector.is_blinking(input_frame.img, left_eye, right_eye)
        problem = self.liveness_detector.validate(input_frame, time_passed)

        if problem:
            self.warning += 'Not live face!'

        return valid

    def eyes_tracker_validation(self, input_frame, img, left_eye, right_eye):
        valid, msg = self.eyes_tracker.check_frame(img, left_eye, right_eye)
        problem = self.eyes_tracker.validate(input_frame, valid)

        if problem:
            self.warning += 'Looked aside!'

        return valid

    def speech_detector_validation(self, input_frame, top_lip, bottom_lip):
        valid = self.speech_detector.is_open(top_lip, bottom_lip)
        problem = self.speech_detector.validate(input_frame, valid)

        if problem:
            self.warning += "Don't speak!"
        return valid

    def face_recognizer_validation(self, input_frame, img, landmarks):
        valid = self.face_recognizer.compare_faces(img, landmarks)
        problem = self.face_recognizer.validate(input_frame, valid)

        if problem:
            self.warning += "Not recognized!"

        return valid

    @staticmethod
    def get_time(time_limit, time_passed):
        time_showing = round(time_limit - time_passed)
        mins = math.trunc(time_showing / 60)
        sec = round(time_showing % 60)
        if mins < 10:
            str_mins = "0" + str(mins)
        else:
            str_mins = str(mins)
        if sec < 10:
            str_sec = "0" + str(sec)
        else:
            str_sec = str(sec)
        return "Time: " + str_mins + ":" + str_sec

    def main_report(self, size, buffer, file_name, fps):
        if len(buffer) != 0:
            file_name = self.test["id_number"] + "_" + file_name
            out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
            for frame in buffer:
                cv2.putText(frame.img, frame.msg, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame.img, frame.time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame.img)
            out.release()
            self.database.add_report(self.student["id"], file_name)

    def start(self, student_id, test_id):
        valid = self.load_student(student_id)
        if valid:
            self.load_test(test_id)
            print(self.test["id_number"] + ": Started.")
            time_limit = self.test["duration"]
            cap = cv2.VideoCapture(0)
            _, input_img = cap.read()
            (h, w) = input_img.shape[:2]
            size = (w, h)
            start = time.time()
            end = False
            while True:
                success, input_img = cap.read()
                if success:
                    time_passed = round(time.time() - start, 1)
                    end = time_passed > time_limit
                    if not end:
                        frame_time = self.get_time(time_limit, time_passed)
                        self.frame_counter = self.frame_counter + 1
                        input_frame = Frame(input_img, frame_time)
                        self.main_buffer.append(input_frame)
                        if self.object_detector_validation(input_frame, h, w):
                            valid, landmarks, landmarks_np = self.face_detector_validation(input_frame, h, w)
                            if valid:
                                image_points = self.face_detector.get_head_pose_landmarks()
                                valid = self.head_pose_detector_validation(input_frame, h, w, image_points)
                                if valid:
                                    left_eye = self.face_detector.get_left_eye_landmarks()
                                    right_eye = self.face_detector.get_right_eye_landmarks()
                                    new_img, landmarks, landmarks_np = self.face_detector.align(input_img, left_eye,
                                                                                                right_eye)
                                    left_eye = self.face_detector.get_left_eye_landmarks()
                                    right_eye = self.face_detector.get_right_eye_landmarks()
                                    valid = self.liveness_detector_validation(input_frame, left_eye, right_eye,
                                                                              int(time_passed))
                                    if valid:
                                        self.eyes_tracker_validation(input_frame, new_img, left_eye, right_eye)
                                    top_lip = self.face_detector.get_top_lip_landmarks()
                                    bottom_lip = self.face_detector.get_bottom_lip_landmarks()
                                    self.speech_detector_validation(input_frame, top_lip, bottom_lip)
                                    self.face_recognizer_validation(input_frame, new_img, landmarks)
                                else:
                                    self.liveness_detector.reset()
                                    self.eyes_tracker.reset()
                                    self.speech_detector.reset()
                                    self.face_recognizer.reset()
                            else:
                                self.head_pose_detector.reset()
                                self.liveness_detector.reset()
                                self.eyes_tracker.reset()
                                self.speech_detector.reset()
                                self.face_recognizer.reset()
                        else:
                            self.face_detector.reset()
                            self.head_pose_detector.reset()
                            self.liveness_detector.reset()
                            self.eyes_tracker.reset()
                            self.speech_detector.reset()
                            self.face_recognizer.reset()

                        if self.warning != "":
                            print("Warning: " + self.warning)
                            self.warning = ""

                        cv2.putText(input_img, frame_time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Test', input_img)
                if cv2.waitKey(1) & 0xFF == ord('q') or end:
                    break

            self.main_report(size, self.main_buffer, "video.avi", 10)
            self.invalid_buffer = [x for x in self.main_buffer if x.valid is False]
            self.main_report(size, self.invalid_buffer, "report.avi", 10)
            print(self.test["id_number"] + ": Finished.")
            # fps = self.frame_counter / time_limit
            # print("FPS: " + str(fps))


proctoring_system = ProctoringSystem()

print("a) Add students")
print("b) Add tests")
print("c) Start test")
action = input()

if action == "a":
    proctoring_system.add_students()
    print("Add students: Done.")
elif action == "b":
    proctoring_system.add_tests()
    print("Add tests: Done.")
else:
    proctoring_system.start("16704", "Math-test2")
    # print("Student id: ")
    # student = input()
    # print("Test id: ")
    # test = input()
    # proctoring_system.start(student, test)
