import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
capture= cv.VideoCapture(0)

mp_hands= mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
hands= mp_hands.Hands(static_image_mode=True,
    max_num_hands= 1, min_detection_confidence= 0.7)

def Min(a, b):
    if a > b:
        return b
    return a

def Max(a, b):
    if a > b:
        return a
    return b

model= joblib.load('model.joblib')

font= cv.FONT_HERSHEY_COMPLEX
fontScale= 1

thickNess= 2

LABEL= {0: "BACKWARD", 1: "FORWARD", 2: "STOP", 3: "TURNLEFT", 4: "TURNRIGHT"}
while True:
    Con, frame= capture.read()
    image_RGB= cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
    
    results= hands.process(image_RGB)
    image_RGB= cv.cvtColor(image_RGB, cv.COLOR_BGR2RGB)
    color= (255, 14, 93)
    thickness= 3
    height, width, depth= image_RGB.shape
    image_RGB= cv.rectangle(image_RGB, (int(0.6*width), int(0.3*height)),
        (width, height), (11, 227, 227), thickness )
    if results.multi_hand_landmarks:
        
        # create 21 landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_RGB, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
        # create rectangle around hand
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
        
        
        image_RGB= cv.rectangle(image_RGB, (int(xMin*width), int(yMin*height)), 
                    (int(xMax*width), int(yMax*height)), color, thickness)
        org= (int(0.6*width), int(0.3*height))
        data= pd.DataFrame(data).values.T
        # print(data) 
        # data= pd.DataFrame(data).values
        # data= np.array(data).reshape(-1, 63)
        y_predict= model.predict(data)
        DAPAN= LABEL[int(y_predict[0])]
        color= (0, 0, 255)
        image_RGB= cv.putText(image_RGB, str(DAPAN), org, font, fontScale, color , thickNess,
                              cv.LINE_AA, False)
        # print(LABEL[int(y_predict[0])])
        # print(data.shape)
        
  
         
    cv.imshow("Image", image_RGB)
    cv.waitKey(50)

