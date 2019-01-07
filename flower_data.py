# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 01:09:37 2019

@author: Bhavesh
"""

import pandas as pd
import numpy as np
import glob
from PIL import Image
#---------Read_dataset--------------

daisy=glob.glob('D:/Image_data/flowers/daisy/*.jpg')
dandelion=glob.glob('D:/Image_data/flowers/dandelion/*.jpg')
rose=glob.glob('D:/Image_data/flowers/rose/*.jpg')
sunflower=glob.glob('D:/Image_data/flowers/sunflower/*.jpg')
tulip=glob.glob('D:/Image_data/flowers/tulip/*.jpg')


#--------Preparing_image_data_with_simple_methods-------------------
def image_data_extractor(image):
    sample_image=Image.open(image)
    sample_image=sample_image.resize((100,100))
    #sample_image=sample_image.convert('L')
    sample_image=np.array(sample_image).flatten()
    return sample_image

#-------Use_function_to_get_dataset------------------

daisy_data=[]
for image in daisy:
    daisy_data.append(list(image_data_extractor(image)))

daisy_data=pd.DataFrame(daisy_data)
daisy_class=pd.DataFrame(list(np.repeat(0,len(daisy))))

dandelion_data=[]
for image in dandelion:
    dandelion_data.append(list(image_data_extractor(image)))

dandelion_data=pd.DataFrame(dandelion_data)
dandelion_class=pd.DataFrame(list(np.repeat(1,len(dandelion))))

rose_data=[]
for image in rose:
    rose_data.append(list(image_data_extractor(image)))

rose_data=pd.DataFrame(rose_data)
rose_class=pd.DataFrame(list(np.repeat(2,len(rose))))

sunflower_data=[]
for image in sunflower:
    sunflower_data.append(list(image_data_extractor(image)))

sunflower_data=pd.DataFrame(sunflower_data)
sunflower_class=pd.DataFrame(list(np.repeat(3,len(sunflower))))


tulip_data=[]
for image in tulip:
    tulip_data.append(list(image_data_extractor(image)))

tulip_data=pd.DataFrame(tulip_data)
tulip_class=pd.DataFrame(np.repeat(4,len(tulip)))

combined_data=pd.concat([daisy_data,dandelion_data,rose_data,sunflower_data,tulip_data],axis=0)
target=pd.concat([daisy_class,dandelion_class,rose_class,sunflower_class,tulip_class],axis=0)
target.columns=['Label']
final_df=pd.concat([combined_data,target],axis=1)

#print(final_df.head())

from sklearn.utils import shuffle
final_df = shuffle(final_df)
#print(final_df.head())

target=final_df[['Label']]
df=final_df.drop('Label',axis=1)

#-----------Split_data_into_train_test-------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,target,test_size=0.20,random_state=42)

#---------Normalize_dataset---------------

X_train=np.array(X_train)/255
X_test=np.array(X_test)/255

#-----------Build_cnn_model------------------
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.activations import relu,softmax
from keras.utils import np_utils

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

#model=Sequential()
#model.add(Dense(70,input_dim=X_train.shape[1],activation='relu'))
#model.add(Dropout(0.20))
#model.add(Dense(40,activation='relu'))
#model.add(Dense(5,activation='softmax'))
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.summary()
#final_model=model.fit(X_train,y_train,batch_size=16,epochs=15,validation_data=(X_test,y_test),verbose=2)

X_train=X_train.reshape(X_train.shape[0],32,32,3)
X_test=X_test.reshape(X_test.shape[0],32,32,3)

#from numpy.random import seed
#seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

model=Sequential()
model.add(Conv2D(64,kernel_size=5,padding='same',input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(128,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(150,kernel_size=2,padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(600,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
final_model=model.fit(X_train,y_train,batch_size=16,epochs=30,validation_data=(X_test,y_test),verbose=2)

import matplotlib.pyplot as plt

accuracy = final_model.history['acc']
val_accuracy = final_model.history['val_acc']
loss = final_model.history['loss']
val_loss = final_model.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



