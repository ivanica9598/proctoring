import cv2
from detectors.eyes_tracker.eye import Eye


class EyesTracker:
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.horizontal_ratio = None

        self.window = []
        self.window_counter = 0
        self.window_limit = 30
        self.window_looking_aside_counter = 0
        self.looking_cons_aside_buffer = []
        self.looking_cons_aside_counter = 0
        self.cons = False

        self.invalid_buffer = []

    def set_new_frame(self, frame, left_eye_landmarks, right_eye_landmarks):
        self.frame = frame
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.eye_left = Eye(frame, left_eye_landmarks)
        self.eye_right = Eye(frame, right_eye_landmarks)
        if self.eye_left.pupils_detected() and self.eye_right.pupils_detected():
            self.horizontal_ratio = (self.eye_left.get_horizontal_percentage() +
                                     self.eye_right.get_horizontal_percentage()) / 2
        return self.horizontal_ratio is not None

    def draw_pupils(self, frame):
        color = (0, 255, 0)
        x_left, y_left = self.eye_left.get_pupil_coordinates()
        x_right, y_right = self.eye_right.get_pupil_coordinates()
        cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
        cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
        cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
        cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

    def check_frame(self, frame, left_eye_landmarks, right_eye_landmarks):
        valid = self.set_new_frame(frame, left_eye_landmarks, right_eye_landmarks)
        if valid:
            # self.draw_pupils(frame)
            # cv2.putText(frame, "Looking aside!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imshow('Eyes', frame)
            if self.horizontal_ratio <= 0.35:
                return False, "Eyes right"
            elif self.horizontal_ratio >= 0.65:
                return False, "Eyes left"
            elif 0.35 < self.horizontal_ratio < 0.65:
                return True, "Eyes center"
        else:
            return True, None

    def reset(self):
        problem = False
        if self.cons:
            if self.looking_cons_aside_counter >= 15:
                for frame in self.looking_cons_aside_buffer:
                    frame.msg += "Looking aside!"
                    self.invalid_buffer.append(frame)
                problem = True
        elif self.window_counter >= 2 / 3 * self.window_limit and self.window_looking_aside_counter >= self.window_counter / 2:
            for i in range(self.window_counter):
                self.window[i].msg += "Looking aside!"
                self.invalid_buffer.append(self.window[i])
            problem = True

        self.window = []
        self.window_counter = 0
        self.window_looking_aside_counter = 0
        self.looking_cons_aside_buffer = []
        self.looking_cons_aside_counter = 0
        self.cons = False

        return problem

    def validate(self, input_frame, valid):
        problem = False

        if self.cons and not valid:
            self.looking_cons_aside_counter = self.looking_cons_aside_counter + 1
            self.looking_cons_aside_buffer.append(input_frame)
            return problem
        elif self.cons:
            self.cons = False
            if self.looking_cons_aside_counter >= 15:
                for frame in self.looking_cons_aside_buffer:
                    frame.msg += "Looking aside!"
                    self.invalid_buffer.append(frame)
                problem = True

        self.window_counter = self.window_counter + 1
        self.window.append(input_frame)

        if valid:
            self.looking_cons_aside_buffer = []
            self.looking_cons_aside_counter = 0
        else:
            self.looking_cons_aside_counter = self.looking_cons_aside_counter + 1
            self.looking_cons_aside_buffer.append(input_frame)
            self.window_looking_aside_counter = self.window_looking_aside_counter + 1

        if self.window_counter == self.window_limit:
            if self.looking_cons_aside_counter > 0:
                self.cons = True
                self.window_counter = self.window_counter - self.looking_cons_aside_counter
                self.window_looking_aside_counter = self.window_looking_aside_counter - self.looking_cons_aside_counter
            else:
                self.cons = False
            if self.window_looking_aside_counter >= self.window_counter / 3:
                for i in range(self.window_counter):
                    self.window[i].msg += "Looking aside!"
                    self.invalid_buffer.append(self.window[i])
                problem = True

            self.window_counter = 0
            self.window_looking_aside_counter = 0
            self.window = []

        return problem

    def get_invalid_buffer(self):
        return self.invalid_buffer
