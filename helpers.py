import numpy as np
import cv2


def shape_to_np(shape, dtype="int"):
    array = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        array[i] = (shape.part(i).x, shape.part(i).y)
    return array


def get_square_box(box):
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:
        return box
    elif diff > 0:
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]


def move_box(box, offset):
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def draw_box(frame, box, class_name, confidence):
    # (startX, startY, endX, endY) = box.astype("int")
    (startX, startY, endX, endY) = box[0], box[1], box[2], box[3]
    label = "{}: {:.2f}%".format(class_name, confidence * 100)
    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# This function accepts a single argument, rect, which is assumed to be a bounding box rectangle produced by a dlib
# detector (i.e., the face detector).

def middle_point(p1, p2):
    x = int((p1.x + p2.x) / 2)
    y = int((p1.y + p2.y) / 2)
    return (x, y)
