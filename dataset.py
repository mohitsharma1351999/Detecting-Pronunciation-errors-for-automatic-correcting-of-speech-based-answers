# for /r %i in (*.mp3) do @ffmpeg -i %i -acodec pcm_s16le -ac 1 -ar 16000 -y converted\%~ni.wav
#for /r %a in (*.mp3) do ffmpeg -v quiet -i "%a" -acodec pcm_s16le -y "%~dpna.wav"
#del /s *.mp3

import shutil
import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("howjsay/bwords") if isfile(join("howjsay/bwords", f))]


for i in onlyfiles:
    folder_name = i.split('.')[-2]
    print("folder_name --->", folder_name)
    destination_folder = "C:/Users/ARSHDEEP SINGH/Desktop/data science/pronunciation errors detection/test/" + folder_name
    source_folder = "howjsay/bwords/" + i
    try:
        os.mkdir(destination_folder)
        shutil.move(source_folder, destination_folder)
    except OSError as error:  
        print(error) 
    
