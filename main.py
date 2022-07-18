import cv2
from face_detector import FaceDetector
from landmarks_detector_1 import LandmarksDetector1
from landmarks_detector_2 import LandmarksDetector2
from face_aligner import FaceAligner
from face_recognizer import FaceRecognizer
from eyes_tracker import EyesTracker
from mouth_tracker import MouthTracker
from head_pose_detector import HeadPoseDetector
from liveness_detector import LivenessDetector
from people_and_devices_detector import PeopleAndDevicesDetector

face_detector = FaceDetector()
landmarks_detector_1 = LandmarksDetector1()
landmarks_detector_2 = LandmarksDetector2()
landmarks_detector = landmarks_detector_2
face_aligner = FaceAligner()
face_recognizer = FaceRecognizer(face_detector, face_aligner)
head_pose_detector = HeadPoseDetector()


# Test face detector
# face_detector.test()

# Test landmarks detector
# landmarks_detector.test(face_detector)

# Test face aligner
# face_aligner.test(face_detector, landmarks_detector_2)

# Test face recognizer
# face_recognizer.test(face_detector, landmarks_detector_2)


# Test head pose detector
# head_pose_detector.test(face_detector, mark_detector_dlib)


def main():
    print('Test started')

    image_path = "images/face.jpg"
    student_image = cv2.imread(image_path)
    student_image_face_boxes, _ = face_detector.find_face_boxes(student_image)

    if len(student_image_face_boxes) == 1:
        student_face_box = student_image_face_boxes[0]
        student_landmarks = landmarks_detector.detect_landmarks(student_image, student_face_box)
        if face_recognizer.set_image(student_image, student_face_box, student_landmarks, True):
            cap = cv2.VideoCapture(0)
            counter = 0
            while True:
                success, input_image = cap.read()
                input_image_face_boxes, _ = face_detector.find_face_boxes(input_image)
                if len(student_image_face_boxes) == 1:
                    input_face_box = input_image_face_boxes[0]
                    face_detector.draw_face(input_image, input_face_box)
                    input_landmarks = landmarks_detector.detect_landmarks(input_image, input_face_box)
                    landmarks_detector.draw_landmarks(input_image, input_landmarks)

                    if counter % 60 == 0:
                        if face_recognizer.set_image(input_image, input_face_box, input_landmarks, False):
                            if face_recognizer.compare_faces():
                                print('Recognized')
                            else:
                                print('Not recognized')
                        else:
                            print('Input image must be recognizable')
                else:
                    print('Input image must have one face!')

                counter = counter + 1
                cv2.imshow('output', input_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print('Student image must be recognizable!')
    else:
        print('Student image must have one face!')


# main()
