import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import io
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

audio_info = pd.read_csv('audio_info.csv')
label = audio_info['lable']
label = list(label)
label = np.array(label)

count = 0
spectograms = []
for file_path in audio_info['file_path']:
    y, sr = librosa.load(file_path)
    count = count + 1
    print("iteration---->", count)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    # print('mel1----->', mel.shape)
    mel = np.transpose(mel, [1, 0])
    # mel = np.expand_dims(mel, -1)
    # print("mel2nd shape----->", mel.shape)
    spectograms.append(mel)
# print(spectograms)
spectograms = np.array(spectograms)
print('done with spectograms')

width_list = []
for i in spectograms:
    width_list.append(i.shape[0])
    # print(i.shape[0])
max_width = max(width_list)

count = 0
padded_spectograms = []
for array in spectograms:
    width = array.shape[0]
    count = count + 1
    print("iteration of pading ---->", count)
    pading_value = max_width - width
    paded_mel = np.pad(array, ((0, pading_value), (0, 0)),'constant', constant_values=(0))
    padded_spectograms.append(paded_mel)
print('done')

padded_spectograms = np.array(padded_spectograms)
padded_spectograms.dump('padded_spectograms.pkl')

print('done with dump')
