from .dataset import NeuroBenchDataset
from torch.utils.data import Dataset
import os
import torch
import math
import numpy as np
import h5py
from scipy.signal import convolve2d

# The spikes recorded in the Primate Reaching datasets have an interval of 4ms.
SAMPLING_RATE = 4e-3


class PrimateReaching(NeuroBenchDataset):
    """
        Dataset for the Primate Reaching Task

        The Dataset can be downloaded from the following website:
        https://zenodo.org/record/583331

        For this task, the following files are selected:
        1. indy_20170131_02.mat
        2. indy_20160630_01.mat
        3. indy_20160622_01.mat
        4. loco_20170301_05.mat
        5. loco_20170217_02.mat
        6. loco_20170210_03.mat

        The description of the structure of the dataset can be found on the website
        in the section: Variable names.

        Once these .mat files are downloaded, store them in the same directory.
    """
    def __init__(self, file_path, filename, num_steps, train_ratio=0.8,
                 mode="3D", model_type="ANN", biological_delay=0,
                 spike_sorting=False, stride=0.004, bin_width=0.028, max_segment_length=2000):
        """
            Initialises the Dataset for the Primate Reaching Task.

            Args:
                file_path (str): The path to the directory storing the matlab files.
                filename (str): The name of the file that will be loaded.
                num_steps (int): number of timesteps the data will be split into.
                train_ratio (float): ratio for how the dataset will be split into training/(val+test) set.
                                     Default is 0.8 (80% of data is training).
                mode (str): The data processed will be either "2D" (data_points, input_features) or 
                            "3D" (data_points, num_steps, input_features). Default is "3D".
                model_type (str): The type of model that will be using the dataset. Currently expects "ANN",
                                  "SNN" or "LSTM" (LSTM to be added later). Default is "ANN"
                biological_delay (int): How many steps of delay is to be applied to the dataset. Default is 0
                                        i.e. no delay applied.
                spike_sorting (bool): Apply spike sorting for processing raw spike data. Default is False.
                stride (float):  How many steps are taken when moving the bin_window. Default is 0.004 (4ms).
                bin_width (float): The size of the bin_window. Default is 0.028 (28ms).
                max_segment_length: Define the upper limits of a segment. Default is 2000 data points (8s)
        """
        # The samples and labels of the dataset
        self.samples = None
        self.labels = None

        # used for input data file management
        self.path = file_path
        self.filename = filename

        # related to processing of spike data
        self.spike_sorting = spike_sorting
        self.delay = biological_delay
        self.stride = stride
        self.bin_width = bin_width
        self.num_steps = num_steps
        self.train_ratio = train_ratio

        # Defines the beginning and end of each segment.
        self.start_end_indices = None
        self.time_segments = None

        # Defines the maximum length of a segment.
        self.max_segment_length = max_segment_length

        # Dataset use mode
        self.mode = mode
        self.model_type = model_type

        # These lists store the index of segments that belongs to training/validation/test set
        self.ind_train, self.ind_val, self.ind_test = [], [], []

        if "indy" in filename:
            self.input_feature_size = 96
        elif "loco" in filename:
            self.input_feature_size = 192
        else:
            raise ValueError("Unexpected filename. Filename should be of either indy or loco")
    
        self.load_data()

        if self.delay > 0:
            self.apply_delay()

        if self.max_segment_length > 0:
            self.remove_segments_by_length()
        
        self.split_data()

    def __len__(self):
        return len(self.ind_train) + len(self.ind_test) + len(self.ind_val)
    
    def __getitem__(self, idx):
        """
            Getter method of the dataloader
        """
        if self.mode == "3D":
            sample = self.get_history(idx)
            label = self.labels[:, idx]
        else:
            sample = self.samples[:, idx]
            label = self.labels[:, idx]
        return sample, label

    def load_data(self):
        """
            Load the data from the matlab file and spike data 
            if spike data has been processed and stored already
        """
        # Assume input is the original dataset, instead of the reconstructed one
        if ".mat" in self.filename:
            file_path = os.path.join(self.path, self.filename)
        else:
            file_path = os.path.join(self.path, self.filename + ".mat")
        dataset = h5py.File(file_path, "r")

        # extract data from datafile
        spikes = dataset["spikes"][()]  # Get the reference object's locations in the HDF5/mat file
        cursor_pos = dataset["cursor_pos"][()]
        target_pos = dataset["target_pos"][()]
        t = np.squeeze(dataset["t"][()])
        new_t = np.arange(t[0] - self.bin_width, t[-1], SAMPLING_RATE)

        # Define the segments' start & end indices
        self.start_end_indices = np.array(self.get_flag_index(target_pos))
        self.time_segments = np.array(self.split_into_segments(self.start_end_indices))

        spike_train = np.zeros((*spikes.shape, len(new_t)), dtype=np.int8)

        # iterate over hdf5 dataframe and preprocess data
        for row_idx, row in enumerate(spikes):
            for col_idx, element in enumerate(row):

                # get indices of spikes and convert data to spike train
                if isinstance(element, np.ndarray):
                    bins, _ = np.histogram(element, bins=new_t.squeeze())
                else:
                    bins, _ = np.histogram(dataset[element][()], bins=new_t.squeeze())

                # histogram is assigns spikes to lower bound of binning window, therefor increment by one to shift to
                # upper bound
                idx = np.nonzero(bins)[0] + 1
                spike_train[row_idx, col_idx, idx] = 1

        if self.spike_sorting:
            # if using spike sorting, reshape # channels x # units into a single dimension => # features
            spike_train = np.transpose(spike_train, (2, 1, 0)).reshape(t.shape[1], -1)

            # remove empty channels
            spike_train = spike_train[:, spike_train.any(axis=0)]
            spike_train = spike_train.transpose()
        else:
            # combine units into channels
            spike_train = np.bitwise_or.reduce(spike_train, axis=0)

        # use convolution to compute binning window
        ratio = int(np.round(self.bin_width / SAMPLING_RATE))
        if ratio != 1:
            binned_spike_train = convolve2d(spike_train, np.ones((1, ratio)), mode='valid')
        else:
            binned_spike_train = spike_train

        # Dimensions: (channels x timesteps)
        self.samples = torch.from_numpy(binned_spike_train).float()
        # Dimensions: (nr_features x timesteps)
        self.labels = torch.from_numpy(cursor_pos).float()

        # convert position to velocity
        self.labels = torch.gradient(self.labels, dim=1)[0]
    
    def apply_delay(self):
        """
            shift the labels by the delay to account for the biological delay between spikes and movement onset
        """
        # Dimension: No_of_Channels*No_of_Records
        self.samples = self.samples[:, :-self.delay]
        self.labels = self.labels[:, self.delay:]

    def split_data(self):
        """
            Split segments into training/validation/test set
        """
        # This is No. of chunks
        split_num = 4
        total_segments = self.time_segments.shape[0]
        sub_length = int(total_segments / split_num)  # This is no of segments in each chunk
        stride = int(self.stride / SAMPLING_RATE)
        # print(total_segments, sub_length)

        train_len = math.floor(self.train_ratio * sub_length)
        val_len = math.floor((sub_length - train_len) / 2)

        offset = int(np.round(self.bin_width / SAMPLING_RATE)) * self.num_steps

        # split the data into 4 equal parts
        # for each part, split the data according to training, testing and validation split
        for split_no in range(split_num):
            for i in range(sub_length):
                # Each segment's Dimension is: No_of_Probes * No_of_Recording
                if i < train_len:
                    self.ind_train += list(np.arange(offset + self.time_segments[split_no * sub_length + i, 0],
                                                     self.time_segments[split_no * sub_length + i, 1], stride))
                elif train_len <= i < train_len + val_len:
                    self.ind_val += list(np.arange(offset + self.time_segments[split_no * sub_length + i, 0],
                                                   self.time_segments[split_no * sub_length + i, 1], stride))
                else:
                    self.ind_test += list(np.arange(offset + self.time_segments[split_no * sub_length + i, 0],
                                                    self.time_segments[split_no * sub_length + i, 1], stride))

    def get_history(self, idx):
        """
            return self.num_steps number of congruent non-overlapping binning windows
        """
        # binning window has a range for "ratio" timesteps
        ratio = int(np.round(self.bin_width / SAMPLING_RATE))

        # compute indices of congruent binning windows
        mask = idx - np.arange(self.num_steps) * ratio
        return self.samples[:, mask]

    def remove_segments_by_length(self):
        """
            remove the segments where its duration exceeds the limit set by
            max_segment_length
        """
        self.time_segments = self.time_segments[self.time_segments[:, 1] - self.time_segments[:, 0] <
                                                self.max_segment_length, :]
        
    def create_dataloader(self, indices, batch_size=256, shuffle=True, drop_last=False):
        """
            Helper method for creating a PyTorch DataLoader based on the split_type.
            Args:
                split_type (str): Defines the split type that will be loaded into the DataLoader.
                                  Can be of the type "Train", "Validation" or "Test".
            :param indices: (list of int) training, testing or validation indices
            :param batch_size: (int) size of batch being processed
            :param shuffle: (boolean) shuffle data
            :param drop_last: (boolean) drop last batch
        """
        current_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(self, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

        return current_loader

    @staticmethod
    def split_into_segments(indices):
        """
            Combine the start and end index into a NumPy array.
        """
        start_end = np.array([indices[:-1], indices[1:]])

        return np.transpose(start_end)

    @staticmethod
    def get_flag_index(target_pos):
        """
            Find where each segment begins and ends
        """
        target_diff = np.diff(target_pos, axis=1, append=target_pos[:, -1].reshape(2, 1))

        indices = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]

        return indices
