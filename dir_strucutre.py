import os
import shutil

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

def class_structure(dst_path, base_path, intersection_list):
    for word_name in intersection_list:
        print('word_name-->', word_name)
        for source in base_path:
            print("scr0---->", source)
            folder_name = word_name.split('.')[-2]  #forming the folder name
            middle_path = str(word_name.split('.')[0][0])+ 'words' +'/'  #traversing to dictionary's sub directory(a,b,c)
            if 'shabadkosh' in source.split('/') :
                for id_number in range(1, 5):
                    filename_id = word_name.split('.')[0] + '_pid' + str(id_number) + '.' + word_name.split('.')[1]
                    scr = source + middle_path + filename_id
                    print("scr---->", scr)
                    dst = dst_path + folder_name + '/'
                    print("just checking", dst) 
                    try:
                        if not os.path.isdir(dst):
                            print('yes')
                            os.mkdir(dst_path+folder_name)
                        shutil.copy(scr, dst)
                    except OSError as error:
                        print(error)

            else:
                scr = source + middle_path + word_name
                print("scr2------>", source)
                dst = dst_path + folder_name +  '/'
                print("just checking2", dst)
                try:
                    if not os.path.isdir(dst):
                        print('yes')
                        os.mkdir(dst)
                    shutil.copy(scr, dst)
                except OSError as error:
                    print(error)
            # shutil.move(src,dst)

intersection = common_words(list_shabadkosh, list_howjsay)
dst_path = "full_intersection/"
base_path = [path_howjsay, path_shabadkosh]
class_structure(dst_path, base_path, intersection)