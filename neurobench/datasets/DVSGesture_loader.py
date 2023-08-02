"""
"""

import torch
from torch.utils.data import DataLoader

from tonic.datasets import DVSGesture as tonic_DVSGesture

from glob import glob

from dataset import NeuroBenchDataset

import os
import stat
import numpy as np

class DVSGesture(NeuroBenchDataset):
    '''
    Requires python 3.10 (introduced root_dir for glob function)
    Installs DVSGesture Dataset with individual events in each file if not yet installed, else pass path of tonic DVSGesture install

    '''
    def __init__(self, path, split='testing'):
        if split == 'training':
            self.dataset = tonic_DVSGesture(save_to=path)
        else:
            self.dataset = tonic_DVSGesture(save_to=path, train = False)

        self.filenames = self.dataset.data
        self.path      = path
        # if split == "testing":
        #     if not installed:
        #         self.dataset = tonic_DVSGesture(save_to=path, train=False)

        #     self.filenames = []
        #     for path, subdirs, files in os.walk(path+'\\ibmGestureTest'):
        #         for name in files:
        #             self.filenames.append(os.path.join(path,name))

        # elif split == "training":
        #     if not installed:
        #         self.dataset = tonic_DVSGesture(save_to=path)

        #     self.filenames = []
        #     for path, subdirs, files in os.walk(path+'\\ibmGestureTraining'):
        #         for name in files:
        #             self.filenames.append(os.path.join(path,name))

        # self.filenames = sorted(list(self.filenames))
        # self.path = path
        # self.labels = sorted(glob("*/", root_dir=path))
        # self.labels = {key[:-1]: idx for idx, key in enumerate(self.labels)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]

    

if __name__ == '__main__':
    dataset = DVSGesture('C:\\Harvard University\\Neurobench\\DVS Gesture\\code\\neurobench\\datasets')
    # print(dataset.filenames)
    # print(len(dataset))
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    print(dataset[0][0].dtype)
    gen_test = DataLoader(dataset,batch_size=4,shuffle=True)
    for local_batch, local_labels in gen_test:
        print(local_batch, local_labels)
    # print(iter(gen_test))
    # print(next(iter(gen_test)))