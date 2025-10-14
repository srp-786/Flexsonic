import cv2
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(1)

mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw=mp.solutions.drawing_utils

ptime=0
ctime=0


while True:
    success, img =cap.read()
    
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                
                height, width,C=img.shape
                cx, cy=int(lm.x*width), int(lm.y*height)
                if id==4:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)


    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img, str(fps), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 2)

    

    # print(results.multi_hand_landmarks)
    cv2.imshow("video", img)
    cv2.waitKey(1)
