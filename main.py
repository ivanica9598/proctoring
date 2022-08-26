import cv2

from detectors.people_detector.people_detector import PeopleDetector
from detectors.face_detector.face_detector import FaceDetector
from detectors.landmarks_detector.landmarks_detector_1 import LandmarksDetector1
from detectors.landmarks_detector.landmarks_detector_2 import LandmarksDetector2
from detectors.face_aligner import FaceAligner
from detectors.mouth_detector.mouth_tracker import MouthTracker
from detectors.eyes_detector.gaze_tracking import GazeTracker
from detectors.liveness_detector.liveness_detector import LivenessDetector
from detectors.face_recognizer.face_recognizer import FaceRecognizer
from detectors.face_recognizer.face_recognizer_2 import FaceRecognizer2
from detectors.head_pose_detector.head_pose_detector import HeadPoseDetector


class ProctoringSystem:
    def __init__(self):
        self.people_detector = PeopleDetector()
        self.face_detector = FaceDetector()
        self.landmarks_detector_1 = LandmarksDetector1()
        self.landmarks_detector_2 = LandmarksDetector2()
        self.landmarks_detector = self.landmarks_detector_2
        self.face_aligner = FaceAligner()
        self.mouth_tracker = MouthTracker()
        self.gaze_tracker = GazeTracker()
        self.liveness_detector = LivenessDetector()
        self.face_recognizer = FaceRecognizer()
        self.head_pose_detector = HeadPoseDetector()

    def test_people_detector(self):
        self.people_detector.test()

    def test_face_detector(self):
        self.face_detector.test()

    def test_landmarks_detector(self):
        self.landmarks_detector.test(self.face_detector)

    def test_face_aligner(self):
        self.face_aligner.test(self.face_detector, self.landmarks_detector)

    def test_mouth_tracker(self):
        image_path = "images/face.jpg"
        student_image = cv2.imread(image_path)
        self.mouth_tracker.test(self.face_detector, self.landmarks_detector, student_image)

    def test_gaze_detector(self):
        self.gaze_tracker.test(self.face_detector, self.landmarks_detector)

    def test_liveness_detector(self):
        self.liveness_detector.test(self.face_detector, self.landmarks_detector)

    def test_face_recognizer(self):
        image_path = "images/face.jpg"
        student_image = cv2.imread(image_path)
        self.face_recognizer.test(self.face_detector, self.landmarks_detector, self.face_aligner, student_image)

    def test_head_pose_detector(self):
        self.head_pose_detector.test(self.face_detector, self.landmarks_detector)

    def start(self):
        image_path = "images/face.jpg"
        student_image = cv2.imread(image_path)
        (student_image_h, student_image_w) = student_image.shape[:2]
        student_image_face_boxes = self.face_detector.detect_faces(student_image, student_image_h, student_image_w)

        if len(student_image_face_boxes) == 1:
            student_face_box = student_image_face_boxes[0][0]
            student_landmarks = self.landmarks_detector.detect_landmarks(student_image, student_face_box)
            if self.face_recognizer.set_image(student_image, student_landmarks, True):
                self.mouth_tracker.set_image(self.landmarks_detector.get_top_lip_landmarks(),
                                             self.landmarks_detector.get_bottom_lip_landmarks(), True)
                cap = cv2.VideoCapture(0)
                counter = 0
                while True:
                    success, input_img = cap.read()
                    if success:
                        (h, w) = input_img.shape[:2]
                        if self.people_detector.detect_people(input_img, h, w):
                            # self.people_detector.draw_people(input_img)
                            input_img_face_boxes = self.face_detector.detect_faces(input_img, h, w)
                            if len(input_img_face_boxes) == 1:
                                input_face_box = input_img_face_boxes[0][0]
                                # self.face_detector.draw_faces(input_img)

                                input_landmarks = self.landmarks_detector.detect_landmarks(input_img, input_face_box)
                                # self.landmarks_detector.draw_landmarks(input_img)

                                new_img = self.face_aligner.align(input_img, h, w,
                                                                  self.landmarks_detector.get_left_eye_landmarks(),
                                                                  self.landmarks_detector.get_right_eye_landmarks())

                                (h, w) = new_img.shape[:2]
                                input_img_face_boxes = self.face_detector.detect_faces(new_img, h, w)
                                input_face_box = input_img_face_boxes[0][0]
                                input_landmarks = self.landmarks_detector.detect_landmarks(new_img, input_face_box)

                                self.mouth_tracker.set_image(self.landmarks_detector.get_top_lip_landmarks(),
                                                             self.landmarks_detector.get_bottom_lip_landmarks(), False)
                                if self.mouth_tracker.compare_faces():
                                    cv2.putText(input_img, 'Talking', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 0, 0), 2)
                                else:
                                    cv2.putText(input_img, 'Not talking', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 0, 0), 2)

                                eyes_check_result = self.gaze_tracker.check_frame(new_img,
                                                                                  self.landmarks_detector.get_left_eye_landmarks(),
                                                                                  self.landmarks_detector.get_right_eye_landmarks())
                                cv2.putText(input_img, eyes_check_result, (90, 60), cv2.FONT_HERSHEY_DUPLEX,
                                            1.6, (147, 58, 31), 2)

                                if counter % 100 == 0:
                                    if self.face_recognizer.set_image(new_img, input_landmarks, False):
                                        if self.face_recognizer.compare_faces():
                                            cv2.putText(input_img, 'Recognized', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                        else:
                                            cv2.putText(input_img, 'Not recognized', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                    else:
                                        print('Input image must be recognizable')
                                counter = (counter + 1) % 101
                            else:
                                print('Input image must have one face!')
                        else:
                            print('Input image must have one person!')

                        counter = counter + 1
                        cv2.imshow('output', input_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                print('Student image must be recognizable!')
        else:
            print('Student image must have one face!')


proctoring_system = ProctoringSystem()

# proctoring_system.test_people_detector()
# proctoring_system.test_face_detector()
# proctoring_system.test_landmarks_detector()
# proctoring_system.test_face_aligner()
# proctoring_system.test_mouth_tracker()
# proctoring_system.test_gaze_detector()
# proctoring_system.test_liveness_detector()
# proctoring_system.test_face_recognizer()
# proctoring_system.test_head_pose_detector()
# proctoring_system.start()
