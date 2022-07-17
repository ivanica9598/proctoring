from face_detector import FaceDetector
from landmarks_detector_1 import LandmarksDetector1
from landmarks_detector_2 import LandmarksDetector2
from face_aligner import FaceAligner
from face_recognizer import FaceRecognizer
from head_pose_detector import HeadPoseDetector
from eyes_tracker import EyesTracker
from liveness_detector import LivenessDetector
from mouth_tracker import MouthTracker
from people_and_devices_detector import PeopleAndDevicesDetector

face_detector = FaceDetector()
landmarks_detector_1 = LandmarksDetector1()
landmarks_detector_2 = LandmarksDetector2()
face_aligner = FaceAligner()
face_recognizer = FaceRecognizer(face_detector, face_aligner)
head_pose_detector = HeadPoseDetector()


# Test face detector
# face_detector.test()

# Test mark detector
# mark_detector.test(face_detector)

# Test mark detector dlib
# mark_detector_dlib.test(face_detector)

# Test face aligner
# face_aligner.test(face_detector, mark_detector)

# Test face recognizer
# face_recognizer.test(face_detector, mark_detector_dlib)

# Test head pose detector
# head_pose_detector.test(face_detector, mark_detector_dlib)


def main():
    print('Program started')


main()
