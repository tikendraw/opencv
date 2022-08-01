import cv2 as cc
import mediapipe as mp
import time

ctime = 0
ptime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cap = cc.VideoCapture(0)

while True:
    success, img = cap.read()
    h, w, c = img.shape
    #convert image to rgb
    imgRGB = cc.cvtColor(img, cc.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            ## if you want to know about coordinates of the points uncomment this
            # for id, lm in enumerate(handLms.landmark):
            #     # print(id, lm)

            #     cx, cy = int(lm.x*w), int(lm.y * h)

            #     ## for highlighting MIDDLEFINGER
            #     # if id in [12,11,10,9]:
            #     #     cc.circle(img, (cx,cy), 11, (255,255,0), cc.FILLED)



            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cc.putText(img, str(int(fps)), (10,70), cc.FONT_HERSHEY_DUPLEX, 3, color = (255,255,0))

    cc.imshow('frame',img)

    cc.waitKey(1)
    