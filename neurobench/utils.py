"""
"""
import numpy as np
import os
import torch
from scipy import signal
import random

def utility_function():
    ...

def find_all_data_file(data_path):
    """
    Find all of the data files of dataset

    :param data_path: File path of dataset
    :return: Names of each data file
    """
    total_data = []
    datalist = os.listdir(data_path)
    for item in datalist:
        if item.endswith('.mat'):
            total_data.append(item)
    return total_data

def save_results(data, train_ratio, delay, fname, path, layer1, layer2):

    with open(os.path.join(path, "test_data.txt"), "a") as f:
        f.write("File_name = " + str(fname) + "\n")
        f.write("Layer1 = " + str(layer1) + " " + "Layer2 = " + str(layer2) + " " + "Train_ratio = " + str(train_ratio) + " " + "Time_shift = " +
                str(delay) + "\n")

        lst_data = ["Training R2: ", str("{0:.4f}".format(data[1])), ";", "Validation R2: ", str("{0:.4f}".format(data[3])), ";", "Testing R2: ", str("{0:.4f}".format(data[5])), ";",
                    "Training loss: ", str("{0:.4f}".format(data[0])), ";", "Validation loss: ", str("{0:.4f}".format(data[2])), ";", "Testing loss: ", str("{0:.4f}".format(data[4]))]
        f.write(" ".join(lst_data) + "\n")

        f.close()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_label = False

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.best_label = False
            if self.counter >= self.patience:
                self.early_stop = True
            else:
                self.early_stop = False
        else:
            self.best_score = score
            self.best_label = True
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #         torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def clip(data, threshold):
    # print(1 - len(torch.nonzero(data))/torch.numel(data))
    data = torch.where(data > threshold, 1.0, 0.0)
    # print(data.dtype)
    # print(1 - len(torch.nonzero(data))/torch.numel(data))
    return data

def butter_lowpass_filter(data, cutoff, order=5):
    b, a = signal.butter(order, cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def bessel_lowpass_filter(data, cutoff, order=4):
    b, a = signal.bessel(order, cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y