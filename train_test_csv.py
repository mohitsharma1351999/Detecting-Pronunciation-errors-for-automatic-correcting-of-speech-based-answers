import os
import shutil
import pandas as pd 
import librosa
import numpy as np

path_howjsay = "full_data/howjsay/"
list_howjsay = []
for word_dir in os.listdir(path_howjsay):
    path_file = path_howjsay + word_dir
    print(path_file)
    L = os.listdir(path_file)
    for i in L:
        if i.endswith('.wav'):
            list_howjsay.append(i.lower())

path_shabadkosh = "full_data/shabadkosh/"
list_shabadkosh = []
for word_dir in os.listdir(path_shabadkosh):
    path_file = path_shabadkosh + word_dir
    print(path_file)
    L = os.listdir(path_file)
    for i in L:
        if i.endswith('.wav'):
            i.split('_')
            only_word = i.split('_')[0]+'.wav'
            list_shabadkosh.append(only_word.lower())

def common_words(list1, list2):
    list1_set = set(list1)
    # print(len(list_set))
    intersection = list1_set.intersection(set(list2))
    # print(intersection)
    intersection = list(intersection)
    intersection.sort()
    return intersection


intersection = common_words(list_shabadkosh, list_howjsay)
base_path = [path_howjsay, path_shabadkosh]
train_path = []
train_label = []

test_path = []
test_label = []
def making_csvs():
    for word_name in intersection:
        if len(word_name.split('.')[0])>=3:
            for source in base_path:
                folder_name = word_name.split('.')[-2]
                middle_path = str(word_name.split('.')[0][0])+ 'words' +'/'
                if 'shabadkosh' in source.split('/') :
                    for id_number in range(1, 5):
                        filename_id = word_name.split('.')[0] + '_pid' + str(id_number) + '.' + word_name.split('.')[1]
                        scr = source + middle_path + filename_id
                        label = str(word_name.split('.')[0])
                        train_path.append(scr)
                        train_label.append(label)
                else:
                    scr = source + middle_path + word_name
                    label = str(word_name.split('.')[0])
                    test_path.append(scr)
                    test_label.append(label)
    return train_path, train_label, test_path, test_label


def sorting_dataframe_by_shape(data_dataframe):
    count = 0
    spectograms_dataframe = []
    for file_path in data_dataframe['file_path']:
        y, sr = librosa.load(file_path)
        count = count + 1
        print("iteration---->", count)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        # print('mel1----->', mel.shape)
        mel = np.transpose(mel, [1, 0])
        # mel = np.expand_dims(mel, -1)
        # print("mel2nd shape----->", mel.shape)
        spectograms_dataframe.append(mel)
    # print(spectograms)
    spectograms_dataframe = np.array(spectograms_dataframe)
    print('done with spectograms dataframe')
    shape_dataframe = []
    for array in spectograms_dataframe:
        shape_dataframe.append(array.shape[0])
    data_dataframe['shape'] = shape_dataframe
    data_dataframe = data_dataframe.sort_values(by=['shape'])
    return data_dataframe


train_path, train_label, test_path, test_label = making_csvs()
# arsh_train_path = train_path[0:400]
# arsh_train_label = train_label[0:400]
# arsh_test_path = test_path[0:100]
# arsh_test_label = test_label[0:100]
train_csv_arsh = pd.DataFrame(data = {'file_path' : train_path, 'lable' : train_label})
sorted_train = sorting_dataframe_by_shape(train_csv_arsh)
sorted_train.to_csv("train_new.csv", index = False)
test_csv_arsh = pd.DataFrame(data = {'file_path' : test_path, 'lable' : test_label})
sorted_test = sorting_dataframe_by_shape(test_csv_arsh)
sorted_test.to_csv("test_new.csv", index = False)
# mohit_train_path = train_path[-400:]
# mohit_train_label = train_label[-400:]
# train_csv_mohit = pd.DataFrame(data = {'file_path' : mohit_train_path, 'lable' : mohit_train_label})
# train_csv_mohit.to_csv("train_mohit.csv", index = False)

# mohit_test_path = test_path[-100:]
# mohit_test_label = test_label[-100:]

# test_csv_mohit = pd.DataFrame(data = {'file_path' : mohit_test_path, 'lable' : mohit_test_label})
# test_csv_mohit.to_csv("test_mohit.csv", index = False)


