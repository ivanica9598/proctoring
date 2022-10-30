import cv2
import time
import math

from database.database import Database
from detectors.object_detector.object_detector import ObjectDetector
from detectors.face_detector.face_detector import FaceDetector
from detectors.head_pose_detector.head_pose_detector import HeadPoseDetector
from detectors.liveness_detector.liveness_detector import LivenessDetector
from detectors.eyes_detector.eyes_detector import EyesDetector
from detectors.speech_detector.speech_detector import SpeechDetector
from detectors.face_recognizer.face_recognizer import FaceRecognizer


class Frame:
    def __init__(self, img, id_num):
        self.img = img
        self.id = id_num
        self.msg = ""


class ProctoringSystem:
    def __init__(self):
        self.database = Database()

        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.head_pose_detector = HeadPoseDetector()
        self.liveness_detector = LivenessDetector()
        self.eyes_detector = EyesDetector()
        self.speech_detector = SpeechDetector()
        self.face_recognizer = FaceRecognizer()

        self.student = None
        self.student_image = None

        self.object_detector_person_buffer = []
        self.object_detector_cellphone_buffer = []
        self.face_detector_buffer = []
        self.head_detector_buffer = []
        self.liveness_detector_buffer = []
        self.eyes_detector_buffer = []
        self.speech_detector_buffer = []
        self.face_recognizer_buffer = []

        self.report_buffer = []
        self.main_buffer = []
        self.frame_counter = 0
        self.warning = ""

    def add_students(self):
        self.database.add_user_to_database("12345", "Petar", "Petrovic", "petar@gmail.com", "images/face1.jpg")
        self.database.add_user_to_database("67890", "Miroslav", "Mikic", "miroslav@gmail.com", "images/face1.jpg")
        self.database.add_user_to_database("16704", "Ivana", "Milivojevic", "ivana@gmail.com", "images/face1.jpg")

    def set_student_image(self, student_id_number):
        self.student, self.student_image = self.database.load_user(student_id_number)
        print(self.student["first_name"] + " " + self.student["last_name"] + ", " + self.student["id_number"])
        (h, w) = self.student_image.shape[:2]
        img = Frame(self.student_image, -1)
        valid, landmarks, landmarks_np = self.face_detector_validation(img, h, w)
        if valid:
            top_lip = self.face_detector.get_top_lip_landmarks()
            bottom_lip = self.face_detector.get_bottom_lip_landmarks()
            self.speech_detector.initialize(top_lip, bottom_lip)
            valid = self.face_recognizer.set_image(self.student_image, landmarks, True)
            return valid
        return False

    def object_detector_validation(self, input_frame, h, w):
        person_valid, cellphone_valid = self.object_detector.detect(input_frame.img, h, w)
        self.object_detector_person_buffer, person_problem = self.object_detector.validate_person(input_frame,
                                                                                                  person_valid,
                                                                                                  self.object_detector_person_buffer)
        self.object_detector_cellphone_buffer, cellphone_problem = self.object_detector.validate_cellphone(input_frame,
                                                                                                           cellphone_valid,
                                                                                                           self.object_detector_cellphone_buffer)
        if person_problem:
            self.warning += 'Not 1 person!'
        if cellphone_problem:
            self.warning += 'Cellphone detected!'

        return person_valid

    def face_detector_validation(self, input_frame, h, w):
        valid, face_boxes = self.face_detector.detect_faces(input_frame.img, h, w)
        self.face_detector_buffer, problem = self.face_detector.validate(input_frame, valid, self.face_detector_buffer)

        if problem:
            self.warning += 'Not 1 face!'
        landmarks = None
        landmarks_np = None
        if valid:
            valid, landmarks, landmarks_np = self.face_detector.detect_landmarks(input_frame.img,
                                                                                 face_boxes[0][0])
        return valid, landmarks, landmarks_np

    def head_pose_detector_validation(self, input_frame, h, w, image_points):
        valid = self.head_pose_detector.detect_head_pose(input_frame.img, h, w, image_points)
        self.head_detector_buffer, problem = self.head_pose_detector.validate(input_frame, valid,
                                                                              self.head_detector_buffer)

        if problem:
            self.warning += 'Head aside!'

        return valid

    def liveness_detector_validation(self, input_frame, left_eye, right_eye, time_passed):
        valid = self.liveness_detector.is_blinking(input_frame.img, left_eye, right_eye)
        self.liveness_detector_buffer, problem = self.liveness_detector.validate(input_frame,
                                                                                 self.liveness_detector_buffer,
                                                                                 time_passed)

        if problem:
            self.warning += 'Not live face!'

        return valid

    def eyes_detector_validation(self, input_frame, img, left_eye, right_eye):
        valid, msg = self.eyes_detector.check_frame(img, left_eye, right_eye)
        self.eyes_detector_buffer, problem = self.eyes_detector.validate(input_frame, valid, self.eyes_detector_buffer)

        if problem:
            self.warning += 'Looked aside!'

        return valid

    def speech_detector_validation(self, input_frame, top_lip, bottom_lip):
        valid = self.speech_detector.is_open(top_lip, bottom_lip)
        self.speech_detector_buffer, problem = self.speech_detector.validate(input_frame, valid,
                                                                             self.speech_detector_buffer)

        if problem:
            self.warning += "Don't speak!"
        return valid

    def face_recognizer_validation(self, input_frame, img, landmarks):
        valid = self.face_recognizer.compare_faces(img, landmarks)
        self.face_recognizer_buffer, problem = self.face_recognizer.validate(input_frame, valid,
                                                                             self.face_recognizer_buffer)

        if problem:
            self.warning += "Not recognized!"

        return valid

    def main_report(self, size, buffer, file_name, fps):
        if len(buffer) != 0:
            buffer = sorted(buffer, key=lambda x: x.id, reverse=False)
            out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
            for frame in buffer:
                cv2.putText(frame.img, frame.msg, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame.img)
            out.release()
            self.database.add_report(self.student["id"], file_name)

    def start(self, time_limit):
        valid = self.set_student_image("16704")
        if valid:
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
                        cv2.putText(input_img, "Time: " + str_mins + ":" + str_sec, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0), 2)
                        self.frame_counter = self.frame_counter + 1
                        input_frame = Frame(input_img.copy(), self.frame_counter)
                        self.main_buffer.append(input_frame)
                        if self.object_detector_validation(input_frame, h, w):
                            valid, landmarks, landmarks_np = self.face_detector_validation(input_frame, h, w)
                            if valid:
                                image_points = self.face_detector.get_head_pose_landmarks()
                                valid = self.head_pose_detector_validation(input_frame, h, w, image_points)
                                if valid:
                                    left_eye = self.face_detector.get_left_eye_landmarks()
                                    right_eye = self.face_detector.get_right_eye_landmarks()
                                    valid, new_img, landmarks, landmarks_np = self.face_detector.align(input_img, h, w,
                                                                                                       left_eye,
                                                                                                       right_eye)
                                    if valid:
                                        left_eye = self.face_detector.get_left_eye_landmarks()
                                        right_eye = self.face_detector.get_right_eye_landmarks()
                                        valid = self.liveness_detector_validation(input_frame, left_eye,
                                                                                  right_eye,
                                                                                  int(time_passed))
                                        if valid:
                                            self.eyes_detector_validation(input_frame, new_img, left_eye,
                                                                          right_eye)
                                        top_lip = self.face_detector.get_top_lip_landmarks()
                                        bottom_lip = self.face_detector.get_bottom_lip_landmarks()
                                        self.speech_detector_validation(input_frame, top_lip, bottom_lip)
                                        self.face_recognizer_validation(input_frame, new_img, landmarks)
                                    else:
                                        self.liveness_detector.reset()
                                        self.eyes_detector_buffer, _ = self.eyes_detector.reset(
                                            self.eyes_detector_buffer)
                                        self.speech_detector_buffer, _ = self.speech_detector.reset(
                                            self.speech_detector_buffer)
                                        self.face_recognizer_buffer, _ = self.face_recognizer.reset(
                                            self.face_recognizer_buffer)
                                else:
                                    self.liveness_detector.reset()
                                    self.eyes_detector_buffer, _ = self.eyes_detector.reset(
                                        self.eyes_detector_buffer)
                                    self.speech_detector_buffer, _ = self.speech_detector.reset(
                                        self.speech_detector_buffer)
                                    self.face_recognizer_buffer, _ = self.face_recognizer.reset(
                                        self.face_recognizer_buffer)
                            else:
                                self.head_detector_buffer, _ = self.head_pose_detector.reset(self.head_detector_buffer)
                                self.liveness_detector.reset()
                                self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                                self.speech_detector_buffer, _ = self.speech_detector.reset(self.speech_detector_buffer)
                                self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)
                        else:
                            self.face_detector_buffer, _ = self.face_detector.reset(self.face_detector_buffer)
                            self.head_detector_buffer, _ = self.head_pose_detector.reset(self.head_detector_buffer)
                            self.liveness_detector.reset()
                            self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                            self.speech_detector_buffer, _ = self.speech_detector.reset(self.speech_detector_buffer)
                            self.face_recognizer_buffer, _ = self.face_recognizer.reset(self.face_recognizer_buffer)

                        if self.warning != "":
                            print(self.warning)
                            cv2.putText(input_img, self.warning, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            self.warning = ""

                        cv2.imshow('Test', input_img)
                if cv2.waitKey(1) & 0xFF == ord('q') or end:
                    break

            self.report_buffer = list(
                set().union(self.object_detector_person_buffer, self.object_detector_cellphone_buffer,
                            self.face_detector_buffer, self.head_detector_buffer,
                            self.eyes_detector_buffer, self.liveness_detector_buffer, self.speech_detector_buffer,
                            self.face_recognizer_buffer))

            self.main_report(size, self.main_buffer, "full_video.avi", 10)
            self.main_report(size, self.report_buffer, "full_report.avi", 10)


proctoring_system = ProctoringSystem()
# proctoring_system.add_students()
proctoring_system.start(240)
