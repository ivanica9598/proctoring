import cv2

from people_counter import PeopleCounter
from face_detector import FaceDetector
from landmarks_detector_1 import LandmarksDetector1
from landmarks_detector_2 import LandmarksDetector2
from face_aligner import FaceAligner
from face_recognizer import FaceRecognizer
from eyes_tracker import EyesTracker
from gaze_tracking import GazeTracking
from mouth_tracker import MouthTracker
from head_pose_detector import HeadPoseDetector
from liveness_detector import LivenessDetector


class ProctoringSystem:
    def __init__(self):
        self.people_counter = PeopleCounter()
        self.face_detector = FaceDetector()
        self.landmarks_detector_1 = LandmarksDetector1()
        self.landmarks_detector_2 = LandmarksDetector2()
        self.landmarks_detector = self.landmarks_detector_2
        self.face_aligner = FaceAligner()
        self.face_recognizer = FaceRecognizer(self.face_detector, self.face_aligner)
        self.mouth_tracker = MouthTracker()
        self.eyes_tracker = EyesTracker()
        self.gaze = GazeTracking()
        self.head_pose_detector = HeadPoseDetector()

    def test_people_counter(self):
        self.people_counter.test()

    def test_face_detector(self):
        self.face_detector.test()

    def test_landmarks_detector(self):
        self.landmarks_detector.test(self.face_detector)

    def test_face_aligner(self):
        self.face_aligner.test(self.face_detector, self.landmarks_detector)

    def test_face_recognizer(self):
        self.face_recognizer.test(self.face_detector, self.landmarks_detector)

    def test_mouth_tracker(self):
        self.mouth_tracker.test(self.face_detector, self.landmarks_detector)

    def test_eyes_tracker(self):
        self.eyes_tracker.test(self.face_detector, self.landmarks_detector)

    def test_gaze_detector(self):
        self.gaze.test(self.face_detector, self.landmarks_detector)

    def test_head_pose_detector(self):
        self.head_pose_detector.test(self.face_detector, self.landmarks_detector)

    def start(self):
        print('Test started')

        image_path = "images/face.jpg"
        student_image = cv2.imread(image_path)
        student_image_face_boxes, _ = self.face_detector.find_face_boxes(student_image)

        if len(student_image_face_boxes) == 1:
            student_face_box = student_image_face_boxes[0]
            student_landmarks = self.landmarks_detector.detect_landmarks(student_image, student_face_box)
            if self.face_recognizer.set_image(student_image, student_face_box, student_landmarks, True):
                self.mouth_tracker.set_image(student_landmarks, True)
                cap = cv2.VideoCapture(0)
                counter = 0
                while True:
                    success, input_image = cap.read()
                    if self.people_counter.detect_persons(input_image):
                        input_image_face_boxes, confidences = self.face_detector.find_face_boxes(input_image)
                        if len(student_image_face_boxes) == 1:
                            input_face_box = input_image_face_boxes[0]
                            self.face_detector.draw_face(input_image, input_face_box, confidences[0])
                            input_landmarks = self.landmarks_detector.detect_landmarks(input_image, input_face_box)
                            self.landmarks_detector.draw_landmarks(input_image)

                            self.mouth_tracker.set_image(input_landmarks, False)
                            if self.mouth_tracker.compare_faces():
                                print('Talking')
                            else:
                                print('Not talking')

                            eyes_check_resulut = self.gaze.check_frame(input_image, input_landmarks)
                            print(eyes_check_resulut)

                            if counter % 60 == 0:
                                if self.face_recognizer.set_image(input_image, input_face_box, input_landmarks, False):
                                    if self.face_recognizer.compare_faces():
                                        print('Recognized')
                                    else:
                                        print('Not recognized')
                                else:
                                    print('Input image must be recognizable')
                        else:
                            print('Input image must have one face!')
                    else:
                        print('Input image must have one person!')

                    counter = counter + 1
                    cv2.imshow('output', input_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                print('Student image must be recognizable!')
        else:
            print('Student image must have one face!')


proctoring_system = ProctoringSystem()

# proctoring_system.test_people_counter()
# proctoring_system.test_face_detector()
# proctoring_system.test_landmarks_detector()
proctoring_system.test_face_aligner()
# proctoring_system.test_face_recognizer()
# proctoring_system.test_mouth_tracker()
# proctoring_system.test_eyes_tracker()
# proctoring_system.test_gaze_detector()
# proctoring_sys#tem.test_head_pose_detector()
# proctoring_system.start()
