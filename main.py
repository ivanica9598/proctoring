import cv2
import time
import webrtcvad
import pyaudio
import threading
import numpy as np
import matplotlib.pyplot as plt

from database.database import Database
from detectors.object_detector.object_detector import ObjectDetector
from detectors.face_detector.face_detector import FaceDetector
from detectors.head_pose_detector.head_pose_detector import HeadPoseDetector
from detectors.liveness_detector.liveness_detector import LivenessDetector
from detectors.gaze_detector.gaze_detector import GazeDetector
from detectors.speech_detector.speech_detector import SpeechDetector
from detectors.face_recognizer.face_recognizer import FaceRecognizer


class Frame:
    def __init__(self, img, frame_time="", time_passed=None):
        self.img = img
        self.msg = ""
        self.time = frame_time
        self.valid = True
        self.time_passed = time_passed
        self.speech = False


class ProctoringSystem:
    def __init__(self):
        self.database = Database()

        self.object_detector = ObjectDetector()
        self.face_detector = FaceDetector()
        self.head_pose_detector = HeadPoseDetector()
        self.liveness_detector = LivenessDetector()
        self.gaze_detector = GazeDetector()
        self.speech_detector = SpeechDetector()
        self.face_recognizer = FaceRecognizer()

        self.student = None
        self.student_image = None
        self.test = None
        self.size = None

        self.invalid_buffer = []
        self.main_buffer = []
        self.frame_counter = 0
        self.warning = ""

        self.voice_dict = {}

    def add_student(self, id_num, first_name, last_name, img_path):
        self.database.add_student(id_num, first_name, last_name, img_path)
        # self.database.add_student("12345", "Petar", "Petrovic", "images/face1.jpg")
        # self.database.add_student("67890", "Miroslav", "Mikic", "images/face1.jpg")
        # self.database.add_student("16704", "Ivana", "Milivojevic", "images/face1.jpg")

    def add_test(self, id_num, h, m, s):
        self.database.add_test(id_num, h, m, s)
        # self.database.add_test("Math-test1", 0, 4, 0)
        # self.database.add_test("Math-test2", 0, 2, 0)
        # self.database.add_test("Math-test3", 3, 3, 0)

    def load_student(self, student_id):
        self.student, self.student_image = self.database.load_student(student_id)
        print("Welcome, " + self.student["first_name"] + " " + self.student["last_name"] + "!")
        print()

        (h, w) = self.student_image.shape[:2]
        frame = Frame(self.student_image)

        valid, _, _ = self.face_detector_validation(frame, h, w)
        if valid:
            left_eye = self.face_detector.get_left_eye_landmarks()
            right_eye = self.face_detector.get_right_eye_landmarks()
            self.student_image, landmarks, _ = self.face_detector.align(self.student_image, left_eye, right_eye)

            top_lip = self.face_detector.get_top_lip_landmarks()
            bottom_lip = self.face_detector.get_bottom_lip_landmarks()
            self.speech_detector.initialize(self.student_image, top_lip, bottom_lip)
            # cv2.imshow('Student', self.student_image)
            valid = self.face_recognizer.set_image(self.student_image, landmarks, True)

        return valid

    def load_test(self, test_id):
        self.test = self.database.load_test(test_id)

    def object_detector_validation(self, input_frame, h, w):
        person_valid, cellphone_valid = self.object_detector.detect(input_frame.img, h, w)
        person_problem = self.object_detector.validate_person(input_frame, person_valid)
        cellphone_problem = self.object_detector.validate_cellphone(input_frame, cellphone_valid)

        if person_problem:
            self.warning += 'More then 1 person!'
        if cellphone_problem:
            self.warning += 'Cellphone detected!'

        return person_valid

    def face_detector_validation(self, input_frame, h, w):
        valid, landmarks, landmarks_np = self.face_detector.detect_faces(input_frame.img, h, w)
        problem = self.face_detector.validate(input_frame, valid)

        if problem:
            self.warning += 'Not 1 person!'

        return valid, landmarks, landmarks_np

    def head_pose_detector_validation(self, input_frame, h, w, image_points):
        valid = self.head_pose_detector.detect_head_pose(input_frame.img, h, w, image_points)
        problem = self.head_pose_detector.validate(input_frame, valid)

        if problem:
            self.warning += "Unallowed head movement detected!"

        return valid

    def liveness_detector_validation(self, input_frame, left_eye, right_eye, time_passed):
        closed = self.liveness_detector.is_blinking(input_frame.img, left_eye, right_eye)
        problem = self.liveness_detector.validate(input_frame, time_passed)

        if problem:
            self.warning += 'Not live face!'

        return closed

    def gaze_detector_validation(self, input_frame, img, left_eye, right_eye, closed):
        valid, msg = self.gaze_detector.check_frame(img, left_eye, right_eye, closed)
        problem = self.gaze_detector.validate(input_frame, valid)

        if problem:
            self.warning += 'Unallowed eye movement detected!'

        return valid

    def speech_detector_validation(self, image, input_frame, top_lip, bottom_lip):
        valid = self.speech_detector.is_open(image, top_lip, bottom_lip)
        problem = self.speech_detector.validate(input_frame, valid)

        if problem:
            self.warning += "Speaking detected!"
        return valid

    def face_recognizer_validation(self, input_frame, img, landmarks):
        valid = self.face_recognizer.compare_faces(img, landmarks)
        problem = self.face_recognizer.validate(input_frame, valid)

        if problem:
            self.warning += "Not recognized!"

        return valid

    def report(self, size, file_name_full, file_name_invalid, fps):
        if len(self.main_buffer) != 0:
            video_path_full = self.test["id_number"] + "_" + file_name_full
            video_path_invalid = self.test["id_number"] + "_" + file_name_invalid
            full_out = cv2.VideoWriter(video_path_full, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
            invalid_out = cv2.VideoWriter(video_path_invalid, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

            for frame in self.main_buffer:
                cv2.putText(frame.img, frame.msg, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame.img, frame.time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                full_out.write(frame.img)
                if not frame.valid:
                    invalid_out.write(frame.img)

            full_out.release()
            invalid_out.release()
            self.database.add_report(self.student["id"], self.test["id_number"], file_name_full, video_path_full)
            self.database.add_report(self.student["id"], self.test["id_number"], file_name_invalid, video_path_invalid)

    def start(self, student_id, test_id):
        valid = self.load_student(student_id)
        if valid:
            self.load_test(test_id)
            print(self.test["id_number"] + ": Started.")
            time_limit = self.test["hours"] * 3600 + self.test["minutes"] * 60 + self.test["seconds"]

            cap = cv2.VideoCapture(0)
            _, input_img = cap.read()
            (h, w) = input_img.shape[:2]
            self.size = (w, h)

            #vad = webrtcvad.Vad(3)
            #sample_rate = 16000
            #frame_duration = 0.02
            #frames_per_buffer = int(frame_duration * sample_rate)

            #FORMAT = pyaudio.paInt16
            #CHANNELS = 1
            #p = pyaudio.PyAudio()
            #stream = p.open(
            #    format=FORMAT,
            #    channels=CHANNELS,
            #    rate=sample_rate,
            #    input=True,
            #    frames_per_buffer=frames_per_buffer
            #)

            start_time = time.time()
            self.video_record(time_limit, start_time, cap, w, h)
            #audio_thread = threading.Thread(target=self.audio_record, args=(
            #    time_limit, start_time, stream, vad, sample_rate, frames_per_buffer))
            #video_thread = threading.Thread(target=self.video_record, args=(time_limit, start_time, cap, w, h))

            #audio_thread.start()
            #video_thread.start()

            #audio_thread.join()
            #video_thread.join()

            cap.release()
            #stream.stop_stream()
            #stream.close()
            #p.terminate()

            #plt.figure(figsize=(15, 5))
            #plt.ylabel('Speech probability')
            #plt.xlabel('Time (s)')
            #plt.xlim(0, time_limit)
            #plt.xticks(np.arange(0, time_limit + 1, 5.0))

            #keys = self.voice_dict.keys()
            #times = []
            #values = []
            #for key in keys:
            #    if self.voice_dict[key] > 0:
            #        values.append(1)
            #        times.append(key)
            #    # else:
            #    #    values.append(0)
            #plt.plot(times, values, c='blue', label="microphone", marker='o', linestyle='None', markersize=2.0)

            #times = []
            #values = []
            #for frame in self.main_buffer:
            #    times.append(frame.time_passed)
            #    if frame.speech:
            #        values.append(1)
            #    else:
            #        values.append(0)

            #plt.plot(times, values, c='red', label="camera")

            #plt.legend()
            #plt.savefig("speech.png")

            #fps = self.frame_counter / time_limit
            #print("FPS: " + str(int(fps)))
            self.report(self.size, "full_video.avi", "report.avi", 15)
            print(self.test["id_number"] + ": Finished.")

    def audio_record(self, time_limit, start_time, stream, vad, sample_rate, frames_per_buffer):
        print("Audio recording: Started.")

        while True:
            time_passed = round(time.time() - start_time, 1)
            if time_passed > time_limit:
                break
            data = stream.read(frames_per_buffer)
            is_speech = vad.is_speech(data, sample_rate)
            if time_passed not in self.voice_dict:
                self.voice_dict[time_passed] = 0
            else:
                if is_speech:
                    self.voice_dict[time_passed] = self.voice_dict[time_passed] + 1
                else:
                    self.voice_dict[time_passed] = self.voice_dict[time_passed] - 1

        print("Audio recording: Stopped. ")

    def video_record(self, time_limit, start_time, cap, w, h):
        print("Video recording: Started.")
        print()
        while True:
            success, input_img = cap.read()
            if success:
                time_passed = round(time.time() - start_time, 1)
                end = time_passed > time_limit
                if end:
                    break
                self.frame_counter = self.frame_counter + 1
                hours, remainder = divmod(round(time_limit - time_passed), 3600)
                minutes, seconds = divmod(remainder, 60)
                frame_time = "Time: " + '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
                input_frame = Frame(input_img, frame_time, time_passed)
                self.main_buffer.append(input_frame)
                if self.object_detector_validation(input_frame, h, w):
                    valid, landmarks, landmarks_np = self.face_detector_validation(input_frame, h, w)
                    if valid:
                        image_points = self.face_detector.get_head_pose_landmarks()
                        valid = self.head_pose_detector_validation(input_frame, h, w, image_points)
                        if valid:
                            left_eye = self.face_detector.get_left_eye_landmarks()
                            right_eye = self.face_detector.get_right_eye_landmarks()
                            new_img, landmarks, landmarks_np = self.face_detector.align(input_img, left_eye, right_eye)
                            left_eye = self.face_detector.get_left_eye_landmarks()
                            right_eye = self.face_detector.get_right_eye_landmarks()
                            closed = self.liveness_detector_validation(input_frame, left_eye, right_eye, time_passed)
                            self.gaze_detector_validation(input_frame, new_img, left_eye, right_eye, closed)
                            top_lip = self.face_detector.get_top_lip_landmarks()
                            bottom_lip = self.face_detector.get_bottom_lip_landmarks()
                            self.speech_detector_validation(new_img, input_frame, top_lip, bottom_lip)
                            self.face_recognizer_validation(input_frame, new_img, landmarks)
                        else:
                            self.liveness_detector.reset()
                            self.gaze_detector.reset()
                            self.speech_detector.reset()
                            self.face_recognizer.reset()
                    else:
                        self.head_pose_detector.reset()
                        self.liveness_detector.reset()
                        self.gaze_detector.reset()
                        self.speech_detector.reset()
                        self.face_recognizer.reset()
                else:
                    self.face_detector.reset()
                    self.head_pose_detector.reset()
                    self.liveness_detector.reset()
                    self.gaze_detector.reset()
                    self.speech_detector.reset()
                    self.face_recognizer.reset()

                if self.warning != "":
                    print("Warning: " + self.warning + " " + str(frame_time))
                    self.warning = ""

                cv2.putText(input_img, frame_time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Test', input_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Video recording: Stopped. ")
        self.object_detector.reset_person()
        self.object_detector.reset_cellphone()
        self.face_detector.reset()
        self.head_pose_detector.reset()
        self.liveness_detector.reset()
        self.gaze_detector.reset()
        self.speech_detector.reset()
        self.face_recognizer.reset()

        cv2.destroyAllWindows()


proctoring_system = ProctoringSystem()

while True:
    print("")
    print("a) Add student")
    print("a) Delete student")
    print("c) Add test")
    print("d) Delete test")
    print("e) Start test")
    action = input()
    print()

    if action == "a":
        print("ID number: ")
        student_id = input()
        print("First name: ")
        student_first_name = input()
        print("Last name: ")
        student_last_name = input()
        print("Image path: ")
        student_img = input()
        proctoring_system.add_student(student_id, student_first_name, student_last_name, student_img)
    elif action == "b":
        print("ID number: ")
        student_id = input()
        proctoring_system.database.delete_student(student_id)
    elif action == "c":
        print("ID number: ")
        test_id = input()
        print("Duration")
        print("Hours: ")
        hours = input()
        print("Minutes: ")
        minutes = input()
        print("Seconds: ")
        seconds = input()
        proctoring_system.add_test(test_id, hours, minutes, seconds)
    elif action == "d":
        print("ID number: ")
        test_id = input()
        proctoring_system.database.delete_test(test_id)
    else:
        proctoring_system.start("16704", "Math-test2")
        #print("Student: ")
        #student = input()
        #print("Test: ")
        #test = input()
        #print()
        #proctoring_system.start(student, test)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        break
