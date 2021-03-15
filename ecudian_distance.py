from scipy.spatial import minkowski_distance
import librosa
import numpy as np

path0 = 'against/against_pid1.wav'
path1 = 'against/human1.wav'
path2 = 'against/against_pid2.wav'
path3 = 'against/alongway.wav'
path4 = 'against/against_pid4.wav'
audio_0, sr_0 = librosa.load(path0)
audio_1, sr_1 = librosa.load(path1)
audio_2, sr_2 = librosa.load(path2)
audio_3, sr_3 = librosa.load(path3)
audio_4, sr_4 = librosa.load(path4)
AVG = []
mfcc_0 = librosa.feature.mfcc(audio_0)
for i in mfcc_0:
    avg = np.mean(i)
    print(avg)
    AVG.append(avg)
print("AVG---->",AVG)

mfcc_1 = librosa.feature.mfcc(audio_1)
AVG1 = []
for i in mfcc_1:
    avg = np.mean(i)
    print(avg)
    AVG1.append(avg)
print("AVG1---->",AVG1)

mfcc_2 = librosa.feature.mfcc(audio_2)
AVG2 = []
for i in mfcc_2:
    avg = np.mean(i)
    print(avg)
    AVG2.append(avg)
print("AVG2---->",AVG2)

mfcc_3 = librosa.feature.mfcc(audio_3)
AVG3 = []
for i in mfcc_3:
    avg = np.mean(i)
    print(avg)
    AVG3.append(avg)
print("AVG3---->",AVG3)

mfcc_4 = librosa.feature.mfcc(audio_4)
AVG4 = []
for i in mfcc_4:
    avg = np.mean(i)
    print(avg)
    AVG4.append(avg)
print("AVG4---->",AVG4)

ecludian_distance = minkowski_distance(AVG, AVG1)
print("eclusian distance between human voice and 1st audio---->", ecludian_distance)
ecludian_distance = minkowski_distance(AVG, AVG2)
print("eclusian distance between different accents---->", ecludian_distance)
ecludian_distance = minkowski_distance(AVG, AVG3)
print("eclusian distance between different words ---->", ecludian_distance)
ecludian_distance = minkowski_distance(AVG1, AVG3)
print("eclusian distance between human1 and alongway ---->", ecludian_distance)