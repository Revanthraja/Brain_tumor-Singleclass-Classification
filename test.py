# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:54:39 2022

@author: Toshiba

"""

import cv2
import numpy as np
import pyttsx3
from keras.models import load_model
from PIL import Image
image_directory='b1/'
model=load_model('BrainTumor10Epoch.h5')
image=cv2.imread('C:/Users/Toshiba/OneDrive/Pictures/7 no.jpg')
img=Image.fromarray(image)

img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img,axis=0)

result=model.predict(input_img)
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def brain():
    if result==0:
        speak(f"This brain is not affected by tumor sir!")
    else:
        speak(f"This brain affected by  tumor sir!")
brain()
    
    
    
print(result)
if result==0:
    print("no brain tumor")
else:
    print("you have brain tumor")
#else:
    #print("car image")