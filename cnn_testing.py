from keras.layers import Dense, Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling2D
import keras
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import io
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
encoded_value = LabelEncoder()

xtrain = pd.read_csv('train_arsh.csv')
ytrain = np.array(list(xtrain['lable']))
ytrain = encoded_value.fit_transform(ytrain)
ytrain_final = np.zeros((ytrain.size, ytrain.max()+1))
ytrain_final[np.arange(ytrain.size), ytrain] = 1
xtest = pd.read_csv('test_arsh.csv')
ytest = np.array(list(xtest['lable']))
ytest = encoded_value.fit_transform(ytest)
ytest_final = np.zeros((ytest.size, ytest.max()+1))
ytest_final[np.arange(ytest.size), ytest] = 1

def mel_spectograms(file_path_list):
    count = 0
    mel_spectograms = []
    for file_path in file_path_list:
        y, sr = librosa.load(file_path)
        count = count + 1
        print("iteration---->", count)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        print('mel1----->', mel.shape)
        mel = np.transpose(mel, [1, 0])
        # mel = np.expand_dims(mel, -1)
        # print("mel2nd shape----->", mel.shape)
        mel_spectograms.append(mel)
    # print(spectograms)
    mel_spectograms = np.array(mel_spectograms)
    print('done with spectograms dataframe')
    return mel_spectograms

def padding(required_width, data):
    count = 0
    padded_array = []
    for array in data:
        width = array.shape[0]
        count = count + 1
        print("iteration of pading ---->", count)
        pading_value = required_width - width
        paded_mel = np.pad(array, ((0, pading_value), (0, 0)),
                        'constant', constant_values=(0))
        padded_array.append(paded_mel)
    padded_array = np.array(padded_array)
    print('done with padding')
    return padded_array

def get_data_from_batches(x, y, b, batch_size=10):
    batch_start = 0
    x = x['file_path'].tolist()
    for file_path in x:
        print('batch_start before increment---->', batch_start)
        batch_end = batch_start + batch_size
        if batch_end > len(x):
            batch_end = len(x)-1
        xarray = mel_spectograms(x[batch_start:batch_end])
        if len(xarray)>1:
            required_width = xarray[len(xarray) -1].shape[0]
            xbatch = padding(required_width, xarray)
            ybatch = y[batch_start:batch_end]
            yield xbatch
            yield ybatch
            yield required_width    
            batch_start += batch_size
            print('batch_start after increment---->', batch_start)

output = get_data_from_batches(xtrain, ytrain_final, 2, batch_size = 10)


# xtrain_mels = mel_spectograms(xtrain['file_path'])
# xtrain_final = padding(380, xtrain_mels)


def get_model():
    model = Sequential()
    input_shape = (21,128)    
    model.add(Conv1D(512, kernel_size=3, strides=1,
                                activation='relu',
                                input_shape=input_shape))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='softmax'))
    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adam(lr=0.0001),
                            metrics=['accuracy'])
    return model
epochs = 30
batch_size = 10
nbbatches = int(len(xtrain)/batch_size)
# model.fit(xtrain_final, ytrain_final,
#                   batch_size=batch_size,
#                   epochs=10,
#                   verbose=1,
#                   validation_split=0.2)

# xtrain = n, w, channels(128)
accuracy_history = []
loss_history = []
val_accuracy_history = []
val_loss_history = []
for e in range(epochs):         # loop for epoch
    # model load
    if e > 0:
        model.load_weights('my_cnn_model')
    for b in range(nbbatches):  # loop for batch
        yield_output = get_data_from_batches(
            xtrain, ytrain_final, b, batch_size = 10)
        xbatch = next(yield_output)
        ybatch = next(yield_output)
        required_width = next(yield_output)
        model = get_model()
        input_shape = (required_width, 128)
        # xbatch should be B, local_max, 128
        # ybatch should be B, 100
        history = model.fit(xbatch, ybatch,
                  batch_size=batch_size,
                  epochs=10,
                  verbose=1,
                  validation_split=0.2)
        model.save_weights('my_cnn_model'.format(epoch=0))
        accuracy_history.append(history.history['accuracy'][0])
        val_accuracy_history.append(history.history['val_accuracy'][0])
        loss_history.append(history.history['loss'])
        val_loss_history.append(history.history['val_loss'])
plt.figure()
print('history--->', history.history.keys())
plt.plot(loss_history)
plt.plot(val_loss_history)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
    # save the model

# use fit in custom way
# write own to create function
# write the log with batch size




