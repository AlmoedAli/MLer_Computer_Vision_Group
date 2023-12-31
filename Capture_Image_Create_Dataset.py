import cv2 as cv
import time
import numpy as np
from pathlib import Path
import mediapipe as mp

#using medipipe library to solve hand_landmarks
mp_hands= mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
hands= mp_hands.Hands(static_image_mode=True,
    max_num_hands= 1, min_detection_confidence= 0.5)

#find Min
def Min(a, b):
    if a > b:
        return b
    return a

#findMax
def Max(a, b):
    if a > b:
        return a
    return b

LABEL= {0: "BACKWARD", 1: "FORWARD", 2: "STOP", 3: "TURNLEFT", 4: "TURNRIGHT"}

#Capture to make dataset has file name .png
def createImageDataSet(labelClassification, hand):
    
    label= labelClassification
    Path('DATASET/'+ label).mkdir(parents= True, exist_ok= True)
    capture= cv.VideoCapture(0)
    
    #factor to show word
    font= cv.FONT_HERSHEY_COMPLEX
    org= (50, 50)
    fontScale= 1
    color= (0, 0, 255)
    thickNess= 2
    if not capture.isOpened():
        print("Camera is not operate!!!!!!")
    else:
        stt= 0
        counter= 0
        while True:
            Con, frame= capture.read()
            # image_RGB= cv.cvtColor(image_RGB, cv.COLOR_BGR2RGB)
            
            if not Con:
                print("Cannot receive frame!!!!")
            else:
                # if counter%2== 0 and counter:  
                cv.imwrite('DATASET/'+ labelClassification+ '/'+  hand + hand + hand + str(counter)+'.png', frame)
                stt+= 1
            
            # add order and labelClassification
            imageShow= cv.putText(cv.flip(frame, 1), str(LABEL[ord(label)- 48]), org, font, fontScale, color, thickNess, cv.LINE_AA, False)
            imageShow= cv.putText(imageShow, str(stt), (250, 50), font, fontScale, (0, 255, 0), thickNess, cv.LINE_AA, False)
            height, width, depth= imageShow.shape
            
            # create rectangle around hand
            # imageShow= cv.rectangle(imageShow, (int(0.6*width), int(0.3*height)),
            #                 (width, height), (11, 227, 227), thickness= 3 )
            
            # convert image into code color BGR2RGB
            image_RGB= cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
            results= hands.process(image_RGB)  # thuan
            if results.multi_hand_landmarks:
                xMax= -np.inf
                yMax= -np.inf
                xMin= np.inf
                yMin= np.inf
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        xMax= Max(xMax, hand_landmarks.landmark[i].x)
                        xMin= Min(xMin, hand_landmarks.landmark[i].x)
                        yMax= Max(yMax, hand_landmarks.landmark[i].y)
                        yMin= Min(yMin, hand_landmarks.landmark[i].y)
            
                # color= (255, 14, 93)
                thickness= 3
                imageShow= cv.rectangle(imageShow, (int(xMin*width), int(yMin*height)), 
                            (int(xMax*width), int(yMax*height)), (0, 255, 0), thickness)
            cv.imshow(str(labelClassification), imageShow)
            cv.waitKey(10)
            if stt > 1000:
                break
            counter+= 1
    capture.release()
    cv.destroyAllWindows()

if __name__== "__main__":
    listDirect= ['2', '3', '4']
    # factor to show word
    font= cv.FONT_HERSHEY_COMPLEX
    org= (100, 100)
    fontScale= 1
    color= (0, 0, 255)
    thickness= 2
    
    handClass= ['left', 'right']
    for direct in listDirect:
        for hand in handClass:
            capture= cv.VideoCapture(0)
            for i in range(7, -1, -1):
            # while True:
                oldTime= np.floor(time.time())
                ret, frame= capture.read()
                frame= cv.flip(frame, 1)
                frame= cv.putText(frame, str(i), org, font, fontScale, color, thickness, cv.LINE_AA, False)
                frame= cv.putText(frame, ' -> '+ str(LABEL[ord(direct)- 48]) +' '+ hand, (120, 100), font, fontScale, (0, 255, 0), thickness, cv.LINE_AA, False )
                
                cv.imshow("CREATE_DATASET", frame)
                cv.waitKey(20)
                while True:
                    newTime= np.floor(time.time())
                    if newTime- oldTime >= 1:
                        break
                    ret, frame= capture.read()
                    frame= cv.flip(frame, 1)
                    frame= cv.putText(frame, str(i), org, font, fontScale, color, thickness, cv.LINE_AA, False)
                    frame= cv.putText(frame, ' -> '+ str(LABEL[ord(direct)- 48])+' '+ hand, (120, 100), font, fontScale, (0, 255, 0), thickness, cv.LINE_AA, False )
                    cv.imshow("CREATE_DATASET", frame)
                    cv.waitKey(20)
            capture.release()
            cv.destroyAllWindows()
            createImageDataSet(direct, hand)