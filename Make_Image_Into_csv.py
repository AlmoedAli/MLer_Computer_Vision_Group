import numpy as np
import cv2 as cv
import mediapipe as mp
import os

mp_hands= mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
hands= mp_hands.Hands(static_image_mode=True,
    max_num_hands= 1, min_detection_confidence= 0.2)

def convertImageCSV(myPath):
    fileCSV= open('dataset.csv', 'a')
    for folder in os.listdir(myPath):
        label= folder
        for file in os.listdir(myPath+ '/'+ folder): 
            file_location= myPath+ '/'+ folder+ '/'+ file
            # print(file_location)
            image= cv.imread(file_location)
            image_RBG= cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
            results= hands.process(image_RBG)
       
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        object= hand_landmarks.landmark[i]
                        fileCSV.write(str(object.x))
                        fileCSV.write(',')
                        fileCSV.write(str(object.y))
                        fileCSV.write(',')
                        fileCSV.write(str(object.z))
                        fileCSV.write(',')
                fileCSV.write(label)
                fileCSV.write('\n')
    fileCSV.close()
    
                    
                    
# if __name__== "__main__":
myPath= 'DATASET'
convertImageCSV(myPath)



            