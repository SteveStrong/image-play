import numpy as np
import cv2

# cams_test = 10
# for i in range(0, cams_test):
#     cap = cv2.VideoCapture(i)
#     test, frame = cap.read()
#     print("i : "+str(i)+" /// result: "+str(test))

cap = cv2.VideoCapture(0)
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame1 = cap.read()

print(frame1.shape)

while (True):
    # Capture frame-by-frame
    ret, frame2 = cap.read()
        
    diff = cv2.absdiff(frame1, frame2)
    frame1 = frame2;
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1200:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

    cv2.imshow('diff',gray)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()