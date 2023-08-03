"""
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from tonic.datasets import DVSGesture as tonic_DVSGesture

from glob import glob

from dataset import NeuroBenchDataset

import os
import stat
import numpy as np
import struct

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
        structured_array = self.dataset[idx][0]

        x_data = np.array(structured_array['x'], dtype = np.int16)
        y_data = np.array(structured_array['y'], dtype = np.int16)
        p_data = np.array(structured_array['p'], dtype = bool)
        t_data = np.array(structured_array['t'], dtype = np.int64)

        return torch.hstack((torch.tensor(x_data), torch.tensor(y_data), torch.tensor(p_data), torch.tensor(t_data))), self.dataset[idx][1]


# from the spiking jelly github:
def load_aedat_v3(file_name: str):
    '''
    :param file_name: path of the aedat v3 file
    :type file_name: str
    :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
    :rtype: Dict
    This function is written by referring to https://gitlab.com/inivation/dv/dv-python . It can be used for DVS128 Gesture.
    '''
    with open(file_name, 'rb') as bin_f:
        # skip ascii header
        line = bin_f.readline()
        while line.startswith(b'#'):
            if line == b'#!END-HEADER\r\n':
                break
            else:
                line = bin_f.readline()

        txyp = {
            't': [],
            'x': [],
            'y': [],
            'p': []
        }
        while True:
            header = bin_f.read(28)
            if not header or len(header) == 0:
                break

            # read header
            e_type = struct.unpack('H', header[0:2])[0]
            e_source = struct.unpack('H', header[2:4])[0]
            e_size = struct.unpack('I', header[4:8])[0]
            e_offset = struct.unpack('I', header[8:12])[0]
            e_tsoverflow = struct.unpack('I', header[12:16])[0]
            e_capacity = struct.unpack('I', header[16:20])[0]
            e_number = struct.unpack('I', header[20:24])[0]
            e_valid = struct.unpack('I', header[24:28])[0]

            data_length = e_capacity * e_size
            data = bin_f.read(data_length)
            counter = 0

            if e_type == 1:
                while data[counter:counter + e_size]:
                    aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                    timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                    x = (aer_data >> 17) & 0x00007FFF
                    y = (aer_data >> 2) & 0x00007FFF
                    pol = (aer_data >> 1) & 0x00000001
                    counter = counter + e_size
                    txyp['x'].append(x)
                    txyp['y'].append(y)
                    txyp['t'].append(timestamp)
                    txyp['p'].append(pol)
            else:
                # non-polarity event packet, not implemented
                pass
        txyp['x'] = np.asarray(txyp['x'])
        txyp['y'] = np.asarray(txyp['y'])
        txyp['t'] = np.asarray(txyp['t'])
        txyp['p'] = np.asarray(txyp['p'])
        return txyp

if __name__ == '__main__':
    dataset = DVSGesture('C:\\Harvard University\\Neurobench\\DVS Gesture\\code\\neurobench\\datasets')
    # print(dataset.filenames)
    # print(len(dataset))
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    print(dataset[0])
    gen_test = DataLoader(dataset,batch_size=2,shuffle=True)
    for local_batch, local_labels in gen_test:
        print(local_batch[0].shape, local_labels)
    # print(iter(gen_test))
    # print(next(iter(gen_test)))