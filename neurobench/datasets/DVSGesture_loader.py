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
import matplotlib.pyplot as plt
# make animation
from matplotlib.animation import FuncAnimation

class DVSGesture(NeuroBenchDataset):
    '''
    Requires python 3.10 (introduced root_dir for glob function)
    Installs DVSGesture Dataset with individual events in each file if not yet installed, else pass path of tonic DVSGesture install
    https://docs.prophesee.ai/stable/tutorials/ml/data_processing/event_preprocessing.html?highlight=metavision_ml%20preprocessing
    event rate: 100K -> dt 1e-5
    '''
    def __init__(self, path, split='testing', data_type = 'frames', preprocessing = 'histo'):
        if split == 'training':
            self.dataset = tonic_DVSGesture(save_to=path)
        else:
            self.dataset = tonic_DVSGesture(save_to=path, train = False)

        self.filenames = self.dataset.data
        self.path      = path
        self.prepr     = preprocessing
        self.data_type = data_type
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
        t_data = np.array(structured_array['t'], dtype = np.int64) # time is in microseconds
        print(t_data[-5:-1])
        xypt = torch.stack((torch.tensor(x_data), torch.tensor(y_data), torch.tensor(p_data), torch.tensor(t_data)),dim = 1)
        if self.data_type == 'frames':
            if self.prepr == 'histo':
                # print(self.dataset[idx][1])
                events = histogram_preprocessing(xypt,delta_t = 5000,h_og = 128, w_og = 128, display_frame=False)
                return events, self.dataset[idx][1]
            
            elif self.prepr == 'stack':
                # print(self.dataset[idx][1])
                events = stack_preprocessing(xypt,delta_t = 5000,h_og = 128, w_og = 128, display_frame=False)
                return events, self.dataset[idx][1]
     
        return xypt, self.dataset[idx][1]
        
        
def stack_preprocessing(xypt, delta_t = 5000, h_og = 128, w_og = 128,channels = 3, display_frame = False):
    tbins = xypt[-1,3]//delta_t
    print(tbins)
    histogram = np.zeros((tbins, channels, h_og, w_og))
    for bin, frame in enumerate(histogram):
        # delete prev neg times
        xypt_new = xypt[xypt[:,3]>=0]
        xypt = xypt_new
        # print(frame.shape)
        # change timestamps
        xypt[:,3] = xypt[:,3] - delta_t
        # print(xypt[0,3],  xypt[0,3] <=0)
        for i in range(len(xypt)):
            if xypt[i,3] <=0:
                if xypt[i,2]==False:
                    frame[0, xypt[i,0],xypt[i,1]] = 255

                else:
                    frame[1, xypt[i,0],xypt[i,1]] = 255
            
                # print(xypt[i,2])

            else:
                # i know this is bad habit, will change later
                continue
           
    if display_frame:
        frame = frame/np.max(frame)

        animation = FuncAnimation(fig, update, frames=tbins,fargs=(histogram,), interval=5)  # Adjust the interval as needed (in milliseconds)
        plt.show()
        
    return histogram

def histogram_preprocessing(xypt, delta_t, h_og, w_og,channels = 3, display_frame = False):
    tbins = xypt[-1,3]//delta_t
    print(tbins)
    histogram = np.zeros((tbins, channels, h_og, w_og))
    for bin, frame in enumerate(histogram):
        # delete prev neg times
        xypt_new = xypt[xypt[:,3]>=0]
        xypt = xypt_new
        # print(frame.shape)
        # change timestamps
        xypt[:,3] = xypt[:,3] - delta_t
        # print(xypt[0,3],  xypt[0,3] <=0)
        for i in range(len(xypt)):
            if xypt[i,3] <=0:
                if xypt[i,2]==False:
                    frame[0, xypt[i,0],xypt[i,1]] = xypt[i,2]+255

                else:
                    frame[1, xypt[i,0],xypt[i,1]] = xypt[i,2]+255
            
                # print(xypt[i,2])

            else:
                # i know this is bad habit, will change later
                continue
           
    if display_frame:
        frame = frame/np.max(frame)
        # fig = plt.figure()
        # fig, ax = plt.subplots()
        # plt.imshow(frame.transpose(1,2,0))
        animation = FuncAnimation(fig, update, frames=tbins,fargs=(histogram,), interval=5)  # Adjust the interval as needed (in milliseconds)
        plt.show()
        
    return histogram
fig, ax = plt.subplots()
def update(frame, frames):
    ax.clear()
    image = frames[frame].transpose(1,2,0)
    # image = image/np.max(image)
    ax.imshow(image, cmap='brg')  # You can adjust the colormap as needed
    ax.set_title(f'Frame {frame}')
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
    path = os.curdir
    dataset = DVSGesture(os.path.join(path,'neurobench/datasets/DVSGesture'))
    # print(dataset.filenames)
    # print(len(dataset))
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    print(dataset[0])
    gen_test = DataLoader(dataset,batch_size=1,shuffle=True)
    for local_batch, local_labels in gen_test:
        print(local_batch[0].shape, local_labels)
    # print(iter(gen_test))
    # print(next(iter(gen_test)))