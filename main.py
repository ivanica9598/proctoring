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


class Frame:
    def __init__(self, img, id_num):
        self.img = img
        self.id = id_num
        self.msg = ""


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

        self.report_buffer = []
        self.main_buffer = []
        self.frame_counter = 0
        self.warning = ""

    def add_students(self):
        self.database.add_user_to_database("12345", "Petar", "Petrovic", "petar@gmail.com", "images/face1.jpg")
        self.database.add_user_to_database("67890", "Miroslav", "Mikic", "miroslav@gmail.com", "images/face1.jpg")
        self.database.add_user_to_database("16704", "Ivana", "Milivojevic", "ivana@gmail.com", "images/face1.jpg")

    def set_student_image(self, student_id_number):
        user, self.student_image = self.database.load_user(student_id_number)
        print(user["first_name"] + " " + user["last_name"] + ", " + user["id_number"])
        (h, w) = self.student_image.shape[:2]
        img = Frame(self.student_image, -1)
        if self.people_detector_validation(img, h, w):
            valid, face_box = self.face_detector_validation(img, h, w)
            if valid:
                valid, landmarks, landmarks_np = self.landmarks_detector.detect_landmarks(self.student_image, face_box)
                if valid:
                    top_lip = self.landmarks_detector.get_top_lip_landmarks()
                    bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                    self.mouth_detector.initialize(top_lip, bottom_lip)
                    valid = self.face_recognizer.set_image(self.student_image, landmarks, True)
                    return valid
        return False

    def people_detector_validation(self, input_frame, h, w):
        valid = self.people_detector.detect_people(input_frame.img, h, w)
        self.people_detector_buffer, problem = self.people_detector.validate(input_frame, valid,
                                                                             self.people_detector_buffer)

        if problem:
            self.warning += 'Not 1 person!'

        return valid

    def face_detector_validation(self, input_frame, h, w):
        valid, face_boxes = self.face_detector.detect_faces(input_frame.img, h, w)
        self.face_detector_buffer, problem = self.face_detector.validate(input_frame, valid, self.face_detector_buffer)

        if problem:
            self.warning += 'Not 1 face!'

        if valid:
            return True, face_boxes[0][0]
        else:
            return False, None

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

    def mouth_detector_validation(self, input_frame, top_lip, bottom_lip):
        valid = self.mouth_detector.is_open(top_lip, bottom_lip)
        self.mouth_detector_buffer, problem = self.mouth_detector.validate(input_frame, valid,
                                                                           self.mouth_detector_buffer)

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
                cv2.putText(frame.img, frame.msg, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame.img)
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
                    self.frame_counter = self.frame_counter + 1
                    input_frame = Frame(input_img.copy(), self.frame_counter)
                    self.main_buffer.append(input_frame)
                    if self.people_detector_validation(input_frame, h, w):
                        valid, face_box = self.face_detector_validation(input_frame, h, w)
                        if valid:
                            valid, landmarks, landmarks_np = self.landmarks_detector.detect_landmarks(input_img,
                                                                                                      face_box)
                            if valid:
                                image_points = self.landmarks_detector.get_head_pose_landmarks()
                                valid = self.head_pose_detector_validation(input_frame, h, w, image_points)
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
                                            valid = self.liveness_detector_validation(input_frame, left_eye, right_eye,
                                                                                      int(time_passed))
                                            if valid:
                                                self.eyes_detector_validation(input_frame, new_img, left_eye, right_eye)
                                            top_lip = self.landmarks_detector.get_top_lip_landmarks()
                                            bottom_lip = self.landmarks_detector.get_bottom_lip_landmarks()
                                            valid = self.mouth_detector_validation(input_frame, top_lip, bottom_lip)
                                            if valid:
                                                self.face_recognizer_validation(input_frame, new_img, landmarks)
                                            else:
                                                self.face_recognizer_buffer, _ = self.face_recognizer.reset(
                                                    self.face_recognizer_buffer)
                                        else:
                                            self.liveness_detector.reset()
                                            self.eyes_detector_buffer, _ = self.eyes_detector.reset(
                                                self.eyes_detector_buffer)
                                            self.mouth_detector_buffer, _ = self.mouth_detector.reset(
                                                self.mouth_detector_buffer)
                                            self.face_recognizer_buffer, _ = self.face_recognizer.reset(
                                                self.face_recognizer_buffer)
                                    else:
                                        self.liveness_detector.reset()
                                        self.eyes_detector_buffer, _ = self.eyes_detector.reset(
                                            self.eyes_detector_buffer)
                                        self.mouth_detector_buffer, _ = self.mouth_detector.reset(
                                            self.mouth_detector_buffer)
                                        self.face_recognizer_buffer, _ = self.face_recognizer.reset(
                                            self.face_recognizer_buffer)
                                else:
                                    self.liveness_detector.reset()
                                    self.eyes_detector_buffer, _ = self.eyes_detector.reset(self.eyes_detector_buffer)
                                    self.mouth_detector_buffer, _ = self.mouth_detector.reset(
                                        self.mouth_detector_buffer)
                                    self.face_recognizer_buffer, _ = self.face_recognizer.reset(
                                        self.face_recognizer_buffer)
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

                    if self.warning != "":
                        print(self.warning)
                        cv2.putText(input_img, self.warning, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.warning = ""
                    cv2.imshow('Test', input_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # new_arr = np.concatenate((self.people_detector_buffer, self.face_detector_buffer, self.head_detector_buffer, self.eyes_detector_buffer, self.liveness_detector_buffer, self.mouth_detector_buffer, self.face_recognizer_buffer), axis=0)
            new_arr = list(set().union(self.people_detector_buffer, self.face_detector_buffer, self.head_detector_buffer, self.eyes_detector_buffer, self.liveness_detector_buffer, self.mouth_detector_buffer, self.face_recognizer_buffer))

            self.main_report(size, self.main_buffer, "full_video.avi", 20)
            self.main_report(size, new_arr, "full_report.avi", 10)


proctoring_system = ProctoringSystem()
# proctoring_system.add_students()
proctoring_system.start()
