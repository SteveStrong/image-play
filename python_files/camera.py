
import numpy as np
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1)
# cap.open(1)

while True:
    success, img = cap.read()
    # imgResult = img.copy()
    # newPoints = findColor(img, myColors,myColorValues)
    # if len(newPoints)!=0:
    #     for newP in newPoints:
    #         myPoints.append(newP)
    # if len(myPoints)!=0:
    #     drawOnCanvas(myPoints,myColorValues)


    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break