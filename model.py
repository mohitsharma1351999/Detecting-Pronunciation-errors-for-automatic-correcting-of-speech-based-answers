from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import pandas as pd
import numpy as np
from data_utils import Data
my_data = Data()


class MY_Model:
    def __init__(self):
        self.input_shape = (None, 128)
        self.batch_size = 10
        self.nbbatches = int(len(my_data.xtrain)/self.batch_size)
        # self.input_shape = (None, 128)

    def get_model(self):
        model = models.Sequential()
        model.add(layers.Conv1D(512, kernel_size=3, strides=1,
                                activation='relu',
                                input_shape=self.input_shape))
        model.add(layers.MaxPooling1D(pool_size=2, strides=2))
        model.add(layers.Conv1D(256, 3, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        # model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='softmax'))
        print(model.summary())
        # training
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])
        return model
