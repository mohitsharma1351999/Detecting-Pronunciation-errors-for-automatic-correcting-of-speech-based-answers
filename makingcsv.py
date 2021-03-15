import os 
import pandas as pd 
path_howjsay = "full_intersection/"
list_howjsay1 = []
list_howjsay = os.listdir(path_howjsay)
list_label = []
list_file_path = []
print(list_howjsay)
for word_dir in list_howjsay:
    file_path = path_howjsay + word_dir
    for file in os.listdir(file_path):
        list_label.append(word_dir)
        list_file_path.append(file_path + '/' + file)

print("list_lable-------->", list_label)
print("list_file_path------>", list_file_path)

audio_info = pd.DataFrame(data = {'file_path' : list_file_path, 'lable' : list_label})
audio_info.to_csv("audio_info.csv", index = False)



 