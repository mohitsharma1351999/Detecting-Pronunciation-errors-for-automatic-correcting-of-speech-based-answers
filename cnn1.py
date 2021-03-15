from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from keras.models import Sequential
from keras.layers import Dense,Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np 
padded_spectograms = np.load('samplepadded_spectograms.pkl', allow_pickle=True)
label = np.load('sample_label.pkl', allow_pickle= True)
# padded_spectograms = np.expand_dims(padded_spectograms, -1)
# label   = np.expand_dims(label, -1)

xtrain, xtest, ytrain, ytest = train_test_split(padded_spectograms, label, test_size=0.30,random_state = 1)
epochs = 10
batch_size = 32
# img_width, img_height = 128, 40
import keras
import keras.utils
input_shape = (380,128)
# input_shape = (img_width, img_height)# 76x126
#input_shape = time(width)(380) *  channels(128)
model = Sequential()
model.add(Conv1D(128, kernel_size = 3, strides=1,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(64, 3, activation='relu'))#change
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(2000, activation='softmax'))
# model.add(Dense(2, activation='sigmoid'))

# training
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
# xtrain = n, w, channels(128)
model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2)