import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
flower_image = cv2.imread("flower.png") #use any image
_, frame = cap.read()
rows, cols, _ = frame.shape
flower_mask = np.zeros((rows, cols), np.uint8)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    flower_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        top_flower = (landmarks.part(29).x, landmarks.part(29).y)
        center_flower = (landmarks.part(30).x, landmarks.part(30).y)
        left_flower = (landmarks.part(31).x, landmarks.part(31).y)
        right_flower = (landmarks.part(35).x, landmarks.part(35).y)

        flower_width = int(hypot(left_flower[0] - right_flower[0],
                           left_flower[1] - right_flower[1]) * 1.7)
        flower_height = int(flower_width * 0.77)

        top_left = (int(center_flower[0] - flower_width / 2),
                              int(center_flower[1] - flower_height / 2))
        bottom_right = (int(center_flower[0] + flower_width / 2),
                       int(center_flower[1] + flower_height / 2))

        flower_new = cv2.resize(flower_image, (flower_width, flower_height))
        flower_new_gray = cv2.cvtColor(flower_new, cv2.COLOR_BGR2GRAY)
        _, flower_mask = cv2.threshold(flower_new_gray, 25, 255, cv2.THRESH_BINARY_INV)

        flower_area = frame[top_left[1]: top_left[1] + flower_height,
                    top_left[0]: top_left[0] + flower_width]
        flower_area_no_flower = cv2.bitwise_and(flower_area, flower_area, mask=flower_mask)
        final_flower = cv2.add(flower_area_no_flower, flower_new)

        frame[top_left[1]: top_left[1] + flower_height,
                    top_left[0]: top_left[0] + flower_width] = final_flower

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27: #escape key
        break
