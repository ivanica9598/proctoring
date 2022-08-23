from scipy.spatial import distance as dist
import cv2

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


class LivenessDetector:

    def __init__(self):
        self.result = None
        self.counter = 0
        self.total = 0

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def test(self, face_detector, landmarks_detector):
        webcam = cv2.VideoCapture(0)

        while True:
            _, frame = webcam.read()
            (h, w) = frame.shape[:2]
            faces = face_detector.detect_faces(frame, h, w)
            landmarks_detector.detect_landmarks(frame, faces[0][0])

            leftEye = landmarks_detector.get_left_eye_landmarks()
            rightEye = landmarks_detector.get_right_eye_landmarks()

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                self.counter += 1
            else:
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    self.total += 1
                self.counter = 0

            cv2.putText(frame, "Blinks: {}".format(self.total), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Liveness detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()
