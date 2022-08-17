import cv2
import os
from PIL import Image
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from  tensorflow.keras.utils import normalize
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense

image_directory='b1/'
dataset=[]
label=[]
no_tumor=os.listdir(image_directory+'no/')
yes_tumor=os.listdir(image_directory+'yes/')
car_tumor=os.listdir(image_directory+'car/')



for i ,image_name in enumerate(no_tumor):
    if (image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
        #print(label)
for i ,image_name in enumerate(yes_tumor):
    if (image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
        #print(label)
for i ,image_name in enumerate(car_tumor):
    if (image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'car/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
        #print(label)
        
        
        
#print(len(label))        
#print(len(dataset))
#print(label)

dataset=np.array(dataset)
label=np.array(label)
#print(dataset)
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.1,random_state=0)
#print(x_train.shape)

#print(y_train.shape)

x_train=normalize(x_train,axis=1)

x_test=normalize(x_test,axis=1)
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



x=model.fit(x_train,y_train,batch_size=10,verbose=1,epochs=15,validation_data=(x_test,y_test),shuffle=False)
model.save('BrainTumor10Epoch.h5')
