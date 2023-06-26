# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:43:12 2022

@author: j6xul
"""

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode = False, max_num_hands= 2, model_complexity=1, min_detection_confidence=0.5,  min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color = (0, 0, 255), thickness = 5)
handConStyle = mpDraw.DrawingSpec(color = (0, 255, 0), thickness = 3)

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    
    if ret:
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        
        # print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                
                for i, lm in enumerate(handLms.landmark):
                    #print(i, lm.x * imgHeight, lm.y * imgWidth, lm.z)
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                    if i == 4:
                        cv2.circle(img, (xPos, yPos), 10, (0, 0, 255), cv2.FILLED)
                        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (0+100, 0+50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('video', img)
    else:
        break
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()