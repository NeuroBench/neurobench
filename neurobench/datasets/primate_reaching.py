"""
"""

from .dataset import Dataset
import os
from scipy.io import loadmat
from neurobench.preprocessing import PreProcessor
import torch
import pickle
import math
import numpy as np

class PrimateReaching(Dataset):
    def __init__(self, hyperparams, path=None, filename=None, postpr_data_path=None, regenerate=False, d_type=torch.float, biological_delay=0,
                 spike_sorting=False, mode="2D", advance=0.016, bin_width=0.208, Np=None, train_ratio=0.8):
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
        self.hyperparams = hyperparams
        self.train_ratio = train_ratio
        self.bin_process = True
        self.start_end_indices = None
        self.time_segments = None
        self.max_segment_length = 0

        self.load_data()

        if self.delay:
            print("Applying Delay to Data")
            self.apply_delay()

        self.split_data()

        if self.mode == "3D":
            self.transform_to_3d()
        

    def __getitem__(self, idx):
        if self.mode == "2D":
            sample = self.samples[:, idx]
            label = self.labels[:, idx]
        elif self.mode == "3D":
            sample = self.samples[idx, :, :]
            label = self.labels[idx, :]
        return sample, label

    def load_data(self):
        file_path = os.path.join(self.path, self.filename + ".mat")
        full_dataset = loadmat(file_path)
        dataset = full_dataset["a"]

        spikes = dataset["spikes"].item()
        t = dataset["t"].item()
        cursor_pos = dataset["cursor_pos"].item()
        target_pos = dataset["target_pos"].item()

        self.start_end_indices = np.array(self.get_flag_index(target_pos))
        self.time_segments = np.array(self.split_into_segments(self.start_end_indices))
        print(self.start_end_indices.shape, self.time_segments.shape)

        try:
            if self.regenerate:
                raise Exception("regenerate postprocessed data...")

            with open(os.path.join(f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl'), 'rb') as f:
                self.samples = pickle.load(f)
                print("Successfully loaded train samples from:", f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl')

            with open(os.path.join(f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl'), 'rb') as f:
                self.labels = pickle.load(f)
                print("Successfully loaded train samples from:", f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl')

            print("Sample shape: {}, Label shape: {}".format(self.samples.shape, self.labels.shape))
            print("spk2 sparsity: {}".format(1 - len(torch.nonzero(self.samples))/torch.numel(self.samples)))

        except:
            self.samples, self.labels = PreProcessor.preprocessing(spikes, t, cursor_pos, self.d_type, spike_sorting=False,
                                                               advance=self.advance, bin_width=self.bin_width,
                                                               Np=self.Np, mode=self.mode)
            print("Sample shape: {}, Label shape: {}".format(self.samples.shape, self.labels.shape))

            if self.filename and self.postpr_data_path:
                os.makedirs(os.path.join(self.postpr_data_path, 'input'), exist_ok=True)
                print("Save postprocessed data:", os.path.join(f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl'))
                with open(os.path.join(f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl'), 'wb') as f:
                    pickle.dump(self.samples, f)

                os.makedirs(os.path.join(self.postpr_data_path, 'label'), exist_ok=True)
                print("Save postprocessed data:", os.path.join(f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl'))
                with open(os.path.join(f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl'), 'wb') as f:
                    pickle.dump(self.labels, f)

            
    def get_flag_index(self, target_pos):
        """
            Find the beginning and end of a segment based on the
            change in value of the target_pos array in the dataset,
            as change in value means that the monkey has reached the target
            and a new target is set.
        """
        target_diff = np.zeros_like(target_pos)

        target_diff[:-1, 0] = np.diff(target_pos[:, 0], n=1)
        target_diff[:-1, 1] = np.diff(target_pos[:, 1], n=1)

        x_pos_ind = np.nonzero(target_diff[:, 0])[0]
        y_pos_ind = np.nonzero(target_diff[:, 1])[0]

        index_union = np.union1d(x_pos_ind, y_pos_ind)

        return index_union

    def split_into_segments(self, indices):
        """
            Each segments start & end is defined as:
            [index[i], index[i+1]), [index[i+1], index[i+2]), ...
        """
        start_end = []
        for i in range(len(indices)-1):
            start_end.append([indices[i], indices[i+1]])

        return start_end

    def bin_processing(self):
        if self.samples.shape[1] != self.labels.shape[1]:
            diff = abs(self.samples.shape[1] - self.labels.shape[1])
            if self.labels.shape[1] - self.samples.shape[1] < 0:
                self.samples = self.samples[:, :-diff]
            elif self.labels.shape[1] - self.samples.shape[1] > 0:
                self.labels = self.labels[:, :-diff]

        advance_num = int(self.advance//0.004)
        bin_width_num = int(self.bin_width//0.004)

        if self.mode == "3D":
            new_sample = torch.zeros((self.samples.shape[0], int(self.samples.shape[1] // advance_num), bin_width_num), dtype=self.d_type)
            new_label = torch.zeros((self.labels.shape[0], int(self.samples.shape[1] // advance_num)), dtype=self.d_type)
            for col in range(new_sample.shape[1]):
                if col <  bin_width_num/advance_num:
                    bin_start = 0
                    bin_end = int(col * advance_num)
                    if col == 0:
                        bin_end = 1
                    new_sample[:, col, bin_start:bin_end] = self.samples[:, bin_start: bin_end]
                else:
                    bin_start = int(col * advance_num - bin_width_num)
                    bin_end = int(col * advance_num)
                    new_sample[:, col, :] = self.samples[:, bin_start: bin_end]

                new_label[:, col] = self.labels[:, col * advance_num]

            if self.Np < bin_width_num:
                sum_num = bin_width_num // self.Np
                new_sample_Np = torch.zeros((new_sample.shape[0], new_sample.shape[1], self.Np), dtype=self.d_type)
                for idx in range(self.Np):
                    start_idx = idx*sum_num
                    end_idx = idx*sum_num + sum_num
                    new_sample_Np[:, :, idx] = torch.sum(new_sample[:, :, start_idx: end_idx], dim=2)
                self.samples = (new_sample_Np > 0).float()
            else:
                self.samples = (new_sample > 0).float()

            self.labels = new_label


        elif self.mode == "2D":
            new_sample = torch.zeros((self.samples.shape[0], int(self.samples.shape[1]// advance_num)), dtype=self.d_type)
            new_label = torch.zeros((self.labels.shape[0], int(self.samples.shape[1]// advance_num)), dtype=self.d_type)
            for col in range(min(new_sample.shape[1], new_label.shape[1])):
                if col <  bin_width_num/advance_num:
                    bin_start = 0
                    bin_end = int(col * advance_num)
                    new_sample[:, col] = torch.sum(self.samples[:, bin_start: bin_end], dim=1)
                else:
                    bin_start = int(col * advance_num - bin_width_num)
                    bin_end = int(col * advance_num)
                    new_sample[:, col] = torch.sum(self.samples[:,  bin_start: bin_end], dim=1)
                new_label[:, col] = self.labels[:, col * advance_num]

            self.samples = new_sample
            self.labels = new_label 
        
        print("Sample shape: {}, Label shape: {}".format(self.samples.shape, self.labels.shape))


    def apply_delay(self):
        if self.delay:
            self.samples = self.samples[:, :-self.delay]
            self.labels = self.labels[:, self.delay:]

    def split_data(self):
        # This is No. of chunks
        split_num = 4
        total_segments = len(self.time_segments)
        sub_length = int(total_segments / split_num) # This is no of segments in each chunk
        print(total_segments, sub_length)

        train_len = math.floor((self.train_ratio) * sub_length)
        val_len = math.floor(0.5 * (sub_length - train_len))
        test_len = sub_length - train_len - val_len
        
        samples = []
        labels = []
        for split_no in range(split_num):
            for i in range(sub_length):
                if i < train_len:
                    samples.append(self.samples[:, self.time_segments[split_no*sub_length + i][0]:self.time_segments[split_no*sub_length + i][1]])
                    labels.append(self.labels[:, self.time_segments[split_no*sub_length + i][0]:self.time_segments[split_no*sub_length + i][1]])
                    self.ind_train += [split_no*sub_length + i]
                elif i >= train_len and i < train_len+val_len:
                    samples.append(self.samples[:, self.time_segments[split_no*sub_length + i][0]:self.time_segments[split_no*sub_length + i][1]])
                    labels.append(self.labels[:, self.time_segments[split_no*sub_length + i][0]:self.time_segments[split_no*sub_length + i][1]])
                    self.ind_val += [split_no*sub_length + i]
                else:
                    samples.append(self.samples[:, self.time_segments[split_no*sub_length + i][0]:self.time_segments[split_no*sub_length + i][1]])
                    labels.append(self.labels[:, self.time_segments[split_no*sub_length + i][0]:self.time_segments[split_no*sub_length + i][1]])
                    self.ind_test += [split_no*sub_length + i]

        self.samples = samples
        self.labels = labels
        print("The final dimension is: ", len(samples), len(labels))

    def transform_to_3d(self):
        advance_num = int(self.advance//0.004)
        bin_width_num = int(self.bin_width//0.004)
        new_samples, new_labels = [], []
        for sample, label in zip(self.samples, self.labels):
            temp_sample = torch.zeros((sample.shape[0], int(sample.shape[1] // advance_num), bin_width_num), dtype=self.d_type)
            temp_label = torch.zeros((label.shape[0], int(sample.shape[1] // advance_num)), dtype=self.d_type)

            for col in range(temp_sample.shape[1]):
                if col <  bin_width_num/advance_num:
                    bin_start = 0
                    bin_end = int(col * advance_num)
                    if col == 0:
                        bin_end = 1
                    temp_sample[:, col, bin_start:bin_end] = sample[:, bin_start: bin_end]
                else:
                    bin_start = int(col * advance_num - bin_width_num)
                    bin_end = int(col * advance_num)
                    temp_sample[:, col, :] = sample[:, bin_start: bin_end]

                temp_label[:, col] = label[:, col * advance_num]

            if self.Np < bin_width_num:
                sum_num = bin_width_num // self.Np
                temp_sample_Np = torch.zeros((temp_sample.shape[0], temp_sample.shape[1], self.Np), dtype=self.d_type)
                for idx in range(self.Np):
                    start_idx = idx*sum_num
                    end_idx = idx*sum_num + sum_num
                    temp_sample_Np[:, :, idx] = torch.sum(temp_sample[:, :, start_idx: end_idx], dim=2)
                new_samples.append((temp_sample_Np > 0).float())
            else:
                new_samples.append((temp_sample > 0).float())

            new_labels.append(temp_label)

        print("New Samples Dim", len(new_samples), new_samples[0].size(), new_samples[10].size())
        print("New Labels Dim", len(new_labels), new_labels[0].size(), new_labels[10].size())
        self.samples = new_samples
        self.labels = new_labels
