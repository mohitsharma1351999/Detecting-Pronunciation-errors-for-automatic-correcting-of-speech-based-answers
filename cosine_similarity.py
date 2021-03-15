from sklearn.metrics.pairwise import cosine_similarity
import pygame
import pyaudio
import wave
import pandas as pd
import numpy as np
import csv
import os
import librosa


def recording(filename="against/recorded.wav", record_seconds=2):
    # the file name output you want to record into
    # filename = "recorded.wav"
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 44100 samples per second
    sample_rate = 44100
    # record_seconds = 5
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()


recording(filename='against/human1.wav')

# calculating the mfccs
pygame.mixer.init()
pygame.mixer.music.load("against/human1.wav")
pygame.mixer.music.play()


header = 'filename'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for filename in os.listdir(f'against/'):
    filename = f'against/{filename}'
    y, sr = librosa.load(filename, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    file = open('dataset.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())

print('cosine similarity between different voices of same word')
dataset = pd.read_csv('dataset.csv')
mfccs = dataset.drop(['filename', 'label'], axis=1)
print(cosine_similarity(mfccs))
