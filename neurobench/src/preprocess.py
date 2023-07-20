import shutil
import argparse

import glob
from metavision_ml.preprocessing.hdf5 import generate_hdf5

DELTA_T=50000
H=360
W=640

def main():
    parser = argparse.ArgumentParser(description='Preprocess dat files')
    parser.add_argument('-dataset', default='train', type=str, help='dataset used train test val')
    args = parser.parse_args()
    
    #input_path = "/app/propheseeMini/mini_dataset/" + args.dataset
    input_path = "/app/prophesee_gen4/prophese_train/" + args.dataset
    input_file = sorted(glob.glob(input_path+'/*.dat'))
    print('length of input:',len(input_file))
    box_labels = sorted(glob.glob(input_path+'/*.npy'))

    #output_folder = "../data/processed/" + args.dataset
    output_folder = "/app/prophesee_gen4/prophese_train/hdf5_files/" + args.dataset

    cnt = 0

    for file_dat in input_file:
        cnt +=1
        print(cnt)
        generate_hdf5(paths=file_dat, output_folder=output_folder, preprocess="multi_channel_timesurface", delta_t=DELTA_T, height=H, width=W,
                start_ts=0, max_duration=None, n_processes = 16)
    for bbox_file in box_labels:
        shutil.copy(bbox_file, output_folder)

if __name__ == '__main__':
    main()
