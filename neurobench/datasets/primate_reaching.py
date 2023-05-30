"""
"""

from .dataset import Dataset
import os
from scipy.io import loadmat
from neurobench.preprocessing import PreProcessor
import torch
import pickle


class PrimateReaching(Dataset):
    def __init__(self, path=None, filename=None, postpr_data_path=None, regenerate=False, d_type=torch.float, biological_delay=0,
                 spike_sorting=False, mode="2D", advance=0.016, bin_width=0.208, Np=None):
        super().__init__()

        self.samples = None
        self.labels = None
        self.spike_sorting = spike_sorting
        self.delay = biological_delay
        self.mode = mode
        self.path = path
        self.postpr_data_path = postpr_data_path
        self.advance = advance
        self.bin_width = bin_width
        self.filename = filename
        self.d_type = d_type
        self.regenerate = regenerate
        self.Np = Np


        # if path is None:
        #     path = select_file()
        #
        # if valid_path(path):
        #     self.path = path

        self.load_data()

        self.apply_delay()

        self.split_data()

    def __getitem__(self, idx):
        if self.mode == "2D":
            sample = self.samples[:, idx]
            label = self.labels[:, idx]
        if self.mode == "3D":
            sample = self.samples[:, idx, :]
            label = self.labels[:, idx, :]
        return sample, label

    def load_data(self):
        file_path = os.path.join(self.path, self.filename + ".mat")
        full_dataset = loadmat(file_path)
        dataset = full_dataset["a"]

        spikes = dataset["spikes"].item()
        t = dataset["t"].item()
        cursor_pos = dataset["cursor_pos"].item()

        try:
            if self.regenerate:
                raise Exception("regenerate postprocessed data...")

            with open(os.path.join(f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl'), 'rb') as f:
                self.samples = pickle.load(f)
                print("Successfully loaded train samples from:", f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl')

            with open(os.path.join(f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl'), 'rb') as f:
                self.labels = pickle.load(f)
                print("Successfully loaded train samples from:", f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl')

        except:
            self.samples, self.labels = PreProcessor.preprocessing(spikes, t, cursor_pos, self.d_type, spike_sorting=False,
                                                               advance=self.advance, bin_width=self.bin_width,
                                                               Np=self.Np, mode=self.mode)


    def apply_delay(self):
        if self.delay:
            self.samples = self.samples[:, :-self.delay]
            self.labels = self.labels[:, self.delay:]



    def split_data(self):
        split_num = 5
        total_len = self.samples.shape[1]
        del_row = round(self.bin_width / self.advance)
        sub_length = int(total_len / split_num)

        train_len = round((0.8) * sub_length)
        val_len = round(0.5 * (sub_length - train_len))
        test_len = sub_length - train_len - val_len

        for num in range(split_num):
            self.ind_train += [x for x in range(num * sub_length + del_row, num * sub_length + train_len)]
            self.ind_val += [x for x in range(num * sub_length + train_len + del_row,
                                              (num * sub_length + train_len) + val_len)]
            self.ind_test += [x for x in range(num * sub_length + train_len + val_len + del_row,
                                            (num * sub_length + train_len + val_len) + test_len)]


