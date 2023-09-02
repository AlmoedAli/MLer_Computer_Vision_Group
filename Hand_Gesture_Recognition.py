import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

LABEL= {0: "BACKWARD", 1: "FORWARD", 2: "STOP", 3: "TURNLEFT", 4: "TURNRIGHT"}
model= joblib.load('model.joblib')
capture= cv.VideoCapture(0)
mp_hands= mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
hands= mp_hands.Hands(static_image_mode=True,
    max_num_hands= 1, min_detection_confidence= 0.5)

font= cv.FONT_HERSHEY_COMPLEX
fontScale= 1
thickness= 2

def Min(a, b):
    if a > b:
        return b
    return a

def Max(a, b):
    if a > b:
        return a
    return b





def handGesture():
    while True:
        Con, frame= capture.read()
        DIRECT= LABEL[2]   
        color= (255, 14, 93)
        # thickness= 3
        image_RGB= cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
        imageShow= cv.cvtColor(image_RGB, cv.COLOR_BGR2RGB)
        height, width, depth= image_RGB.shape
        results= hands.process(image_RGB)
        if results.multi_hand_landmarks:
            # create 21 landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(imageShow, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
            # find x, y, and getData from capture
            data= []
            xMax= -np.inf
            yMax= -np.inf
            xMin= np.inf
            yMin= np.inf
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):                  
                    data.append(hand_landmarks.landmark[i].x)
                    data.append(hand_landmarks.landmark[i].y)
                    data.append(hand_landmarks.landmark[i].z)
                    xMax= Max(xMax, hand_landmarks.landmark[i].x)
                    xMin= Min(xMin, hand_landmarks.landmark[i].x)
                    yMax= Max(yMax, hand_landmarks.landmark[i].y)
                    yMin= Min(yMin, hand_landmarks.landmark[i].y)
            
            #create rectangle around hand     
            imageShow= cv.rectangle(imageShow, (int(xMin*width), int(yMin*height)), 
                        (int(xMax*width), int(yMax*height)), color, thickness)
            
            # Guess direct
            data= pd.DataFrame(data).values.T
            y_predict= model.predict(data)
            proba= model.predict_proba(data)
            for i in range (0, 5, 1):
                if proba[0][i] > 0.5:
                    DIRECT= LABEL[int(y_predict[0])]
                    break
            org= (int(xMin*width), int(yMin*height)- 5)
            color= (0, 0, 255)
            fontScaleIf= 0.65
            thicknessIf= 2
            imageShow= cv.putText(imageShow, str(DIRECT), org, font, fontScaleIf, color, thicknessIf, cv.LINE_AA, False)     
            cv.imshow("Image", imageShow)
            cv.waitKey(20)

        else:
            org= (int(0.6*width), int(0.3*height))
            color= (0, 0, 255)
            imageShow= cv.putText(imageShow, str(DIRECT), org, font, fontScale, color, thickness, cv.LINE_AA, False)     
            cv.imshow("Image", imageShow)
            cv.waitKey(20)

if __name__== '__main__':
    handGesture()
