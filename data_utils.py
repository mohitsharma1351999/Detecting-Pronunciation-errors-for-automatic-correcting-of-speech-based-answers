from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
encoded_value = LabelEncoder()
import numpy as np 
import librosa
import pandas as pd 

xtrain = pd.read_csv('train_arsh.csv')
ytrain = np.array(list(xtrain['lable']))
ytrain = encoded_value.fit_transform(ytrain)
onehot_encoder = OneHotEncoder(sparse=False)
ytrain = ytrain.reshape(len(ytrain), 1)
ytrain_final = onehot_encoder.fit_transform(ytrain)
xtest = pd.read_csv('test_arsh.csv')
ytest = np.array(list(xtest['lable']))
ytest = encoded_value.fit_transform(ytest)
ytest_final = np.zeros((ytest.size, ytest.max()+1))
ytest_final[np.arange(ytest.size), ytest] = 1

class Data:
    def __init__(self):
        self.xtrain = pd.read_csv('train_arsh.csv')
        self.ytrain = ytrain_final
        print('inside the data class ----->', self.ytrain.shape, self.ytrain[0])
        self.xtest = pd.read_csv('test_arsh.csv')
        self.ytest = ytest_final

    def mel_spectograms(self, file_path_list):
        count = 0
        mel_spectograms = []
        for file_path in file_path_list:
            y, sr = librosa.load(file_path)
            count = count + 1
            # print("iteration---->", count)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            # print('mel1----->', mel.shape)
            mel = np.transpose(mel, [1, 0])
            # mel = np.expand_dims(mel, -1)
            # print("mel2nd shape----->", mel.shape)
            mel_spectograms.append(mel)
        # print(spectograms)
        mel_spectograms = np.array(mel_spectograms)
        print('done with spectograms dataframe')
        return mel_spectograms
    
    def padding(self, required_width, data):
        count = 0
        padded_array = []
        for array in data:
            width = array.shape[0]
            count = count + 1
            # print("iteration of pading ---->", count)
            pading_value = required_width - width
            paded_mel = np.pad(array, ((0, pading_value), (0, 0)),
                            'constant', constant_values=(0))
            padded_array.append(paded_mel)
        print('done with padding')
        return np.array(padded_array)

    
    def get_data_from_batches(self, x, y, batch_size=10):
        batch_start = 0
        x = x['file_path'].tolist()
        for file_path in x:
            print('batch_start before increment---->', batch_start)
            batch_end = batch_start + batch_size
            if batch_end > len(x):
                batch_end = len(x)-1
            xarray = self.mel_spectograms(x[batch_start:batch_end])
            if len(xarray)>1:
                required_width = xarray[len(xarray) -1].shape[0]
                xbatch = self.padding(required_width, xarray)
                ybatch = y[batch_start:batch_end]
                # print('ybatch type--->', type(ybatch))
                yield xbatch, ybatch
                # yield ybatch
                # yield required_width    
                batch_start += batch_size
                print('batch_start after increment---->', batch_start)