# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 19:57:03 2022

@author: j6xul
"""

import cv2
import numpy as np
import random

'''
# read image
img = cv2.imread('pic.jpg')
# resize image
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
print(type(img))
print(img.shape)
#print(img) # BGR

# show image
cv2.imshow('img', img)
# control window
cv2.waitKey(0)
'''

'''
# read video = many image
cap = cv2.VideoCapture('video.mp4')
#cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read() # bool, img
    if ret:
        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        cv2.imshow('video', frame)
    else:
        break
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''

'''
# creat img
#img = np.empty((300, 300, 3), np.uint8) # array size, pixel size
# read image
img = cv2.imread('pic.jpg')
new_img = img[:150, 200:400]
# resize image
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
for row in range(300):
    for col in range(img.shape[1]):
        img[row][col] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
cv2.imshow('img', img)
cv2.imshow('img2', new_img)
cv2.waitKey(0)
'''

'''
# modify img
img = cv2.imread('pic.jpg')
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
kernel = np.ones((3, 3), np.uint8)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img, (7, 7), 10)
canny = cv2.Canny(img, 100, 150)
dilate = cv2.dilate(canny, kernel, iterations = 1)
img2 = cv2.erode(dilate, kernel, iterations = 1)

cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.waitKey(0)
'''

'''
# add lines
img = np.zeros((600, 600, 3), np.uint8)
cv2.line(img, (0, 0), (200, 300), (255, 255, 255), 3)
cv2.rectangle(img, (100, 100), (200, 500), (255, 255, 255), 3)
cv2.rectangle(img, (300, 100), (400, 400), (255, 255, 255), cv2.FILLED)
cv2.circle(img, (250, 250), 50, (255, 0, 0), 3)
cv2.circle(img, (250, 250), 10, (255, 0, 0), cv2.FILLED)
cv2.putText(img, 'Hello world', (100, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

cv2.imshow('img', img)
cv2.waitKey(0)
'''

'''
# detect color
img = cv2.imread('pic.jpg')
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)

def empty(v):
    pass

cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 640, 320)
cv2.createTrackbar('Hue Min', 'TrackBar', 0, 179, empty)
cv2.createTrackbar('Hue Max', 'TrackBar', 179, 179, empty)
cv2.createTrackbar('Sat Min', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Sat Max', 'TrackBar', 255, 255, empty)
cv2.createTrackbar('Val Min', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Val Max', 'TrackBar', 255, 255, empty)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBar')
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBar')
    s_min = cv2.getTrackbarPos('Sat Min', 'TrackBar')
    s_max = cv2.getTrackbarPos('Sat Max', 'TrackBar')
    v_min = cv2.getTrackbarPos('Val Min', 'TrackBar')
    v_max = cv2.getTrackbarPos('Val Max', 'TrackBar')
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('img', img)
    cv2.imshow('img2', mask)
    cv2.imshow('img3', result)
    cv2.waitKey(1)
'''
 
'''
# detect line
img = cv2.imread('shape.jpg')
img2 = img.copy()
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img, 150, 200)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    cv2.drawContours(img2, cnt, -1, (255, 0, 0), 3)
    area = cv2.contourArea(cnt) # surface area
    if area > 500:
        peri = cv2.arcLength(cnt, True) # surface length
        vertices = cv2.approxPolyDP(cnt, peri * 0.02, True)
        corners = len(vertices)
        x, y, w, h = cv2.boundingRect(vertices)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 4)
        if corners == 3:
            cv2.putText(img2, 'triangle', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif corners == 4:
            cv2.putText(img2, 'quadrilateral', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif corners == 5:
            cv2.putText(img2, 'pentagon', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif corners >= 6:
            cv2.putText(img2, 'circle', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.waitKey(0)
'''

'''
# detect face

img = cv2.imread('lenna.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('face_detect.xml')
faceRect = faceCascade.detectMultiScale(gray, 1.1, 5)
print(len(faceRect))

for (x, y, w, h) in faceRect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
'''


# draw
#cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (0,0), fx = 1.4, fy = 1.4)
        cv2.imshow('video', frame)
    else:
        break
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()








