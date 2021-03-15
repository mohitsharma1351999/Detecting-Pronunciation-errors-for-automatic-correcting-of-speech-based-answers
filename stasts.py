import eyed3
import pprint
import os
import csv

root_dir = r"C:\Users\ARSHDEEP SINGH\Desktop\scrapy\howjsay"
root_dir2 = "full_data/shabadkosh"

csvfile = open('time2.csv', 'w', newline='')
fieldnames = ['name', 'duration']
csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
csv_writer.writeheader()

for path in os.listdir(root_dir2):
    count = 0
    current_dir = root_dir2 + '/' + path
    print('current_dir------->', current_dir)
    if os.path.isdir(current_dir):
        print("path---->", current_dir)
        for file in os.listdir(current_dir):
            file_path = current_dir + '/' + file
            if os.path.isfile(file_path):
                audioFile = eyed3.load(file_path)
                if audioFile is not None:
                    time_in_secs = audioFile.info.time_secs
                    time_in_milli = time_in_secs * 1000
                    row_data = {"name": file, "duration": time_in_milli}
                    print("row_data----->", row_data)
                    csv_writer.writerow(row_data)
                count += 1
        print(count)
