
import os
import sys


import glob
from metavision_ml.preprocessing.hdf5 import generate_hdf5


input_path = "" # the folder of unpacked .dat file
input_file = sorted(glob.glob(input_path+'/*.dat'))
print('length of input:',len(input_file))
box_labels = sorted(glob.glob(input_path+'/*.npy'))

output_folder = ""  #the folder of output path
# hint: remember to copy labels to the output folder

cnt = 0

for file_dat in input_file:
    cnt +=1
    print(cnt)
    generate_hdf5(paths=file_dat, output_folder=output_folder, preprocess="multi_channel_timesurface", delta_t=50000, height=360, width=640,
            start_ts=0, max_duration=None, n_processes = 16)

