# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:22:07 2025

@author: Admin
"""

import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import os
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from keras.optimizers import Adam

LE = LabelEncoder()
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] #define how many classes we have ---> will be 20 for CW


#=========== Vars ================
dataInput = []
data = []
labels = []

#========== File Parsing =========

files = sorted(glob.glob('mfccs/*.npy'))
dataInput = [np.load(f) for f in files] #get temp array to feed into loop so max can be calculated before it enters loop
maxFrames = max(m.shape[1] for m in dataInput) #find max frame size which in labsheet is 40,21

for f, m in zip(files, dataInput):
    m = np.pad(m, ((0,0), (0, maxFrames-m.shape[1]))) #pad all frames to the shape of 40,21 so all are same size
    data.append(m) #data array will store padded data
    stemFilename = (Path(os.path.basename(f)).stem) 
    label = stemFilename.split('_')
    labels.append(label[0]) #derive labels from filenames for each file

print(data[0].shape) #sanity check for padding

data = data / np.max(data) #stack all 500 padded MFCC's along sample axis (shape is now slice/mfcc/timeframes)
LE = LE.fit(classes) #map classes to labels
labels=to_categorical(LE.transform(labels)) #one hot encoding for cross entropy
x_train, x_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0) #partition training data at 80%
x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=0) #partition validation and training as final 20% (10% val, 10% test)

#============== Create the model ================

def create_model():
    numClasses=10 #define classes
    model=Sequential()
    model.add(InputLayer(input_shape=(40, 21, 1))) #expects frequency, time and channel
    model.add(Conv2D(64, (3, 3), activation='relu')) #learn 3x3 time frequency patterns
    model.add(MaxPooling2D(pool_size=(3, 3))) #downsampling to reduce params
    model.add(Flatten()) #flattens 2d feature vectors into 1d vector
    model.add(Dense(256)) #learn the higher level features
    model.add(Activation('relu')) 
    model.add(Dense(numClasses)) #class probabilities for digits 0-9
    model.add(Activation('softmax'))
    return model

model = create_model() 
model.compile(loss='categorical_crossentropy', #set loss model to categorical cross entropy to work with one hot labels
metrics=['accuracy'], optimizer=Adam(learning_rate=0.01)) #set learning rate here i think?
#model.load_weights('digit_classification.weights.h5') #uncomment if you want to save the model weights
model.summary() 

#=========== Hyper-parameters to tweak ============

num_epochs = 11 #amount of full passes through data you conduct
num_batch_size = 32 #amount of data batches fed in per epoch

history = model.fit(x_train, y_train, validation_data=(x_val,
y_val), batch_size=num_batch_size, epochs=num_epochs,
verbose=1) #this i believe tracks performance at each epoch

#============ Testing and plotting ================

#test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
#print(f"Restored model accuracy: {test_acc:.4f}")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#model.save_weights('digit_classification.weights.h5')
#print("Model weights saved successfully!")


