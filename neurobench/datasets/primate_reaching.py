"""
"""

from .dataset import Dataset
import os
import torch
from torch.utils.data import DataLoader
import math
import numpy as np
import h5py
from scipy.signal import convolve2d

DATA_FREQ = 4e-3


class PrimateReaching(Dataset):
    def __init__(self, hyperparams, biological_delay=0, spike_sorting=False, mode="2D",
                 stride=0.004, bin_width=0.208, model_type='ANN', max_segment_len=2000):
        super().__init__()

        self.samples = None
        self.labels = None
        self.spike_sorting = spike_sorting
        self.delay = biological_delay
        self.mode = mode
        self.path = hyperparams['dataset_file']
        self.stride = stride
        self.bin_width = bin_width
        self.filename = hyperparams['filename']
        self.num_steps = hyperparams['num_steps']
        self.hyperparams = hyperparams
        self.train_ratio = hyperparams['Train_data_ratio']
        self.start_end_indices = None
        self.time_segments = None
        self.model_type = model_type
        self.max_segment_length = max_segment_len

        self.load_data()

        # Remove segments too long
        self.remove_segments_by_length()

        # shift labels according to delay
        if self.delay:
            self.apply_delay()

        # split the data into training, testing and validation data
        self.split_data_vanilla()

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

    def __len__(self):
        return self.samples.shape[1]

    def load_data(self):
        """
            Load the data and bin it according to binning window and stride
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
        new_t = np.arange(t[0] - self.bin_width, t[-1], DATA_FREQ)

        # Define the segments' start & end indices
        self.start_end_indices = np.array(self.get_flag_index(target_pos))
        self.time_segments = np.array(self.split_into_segments(self.start_end_indices))
        print("{} Segments with first one {}-{}".format(len(self.time_segments), self.time_segments[0][0],
                                                        self.time_segments[0][1]))

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
        ratio = int(np.round(self.bin_width / DATA_FREQ))
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

    def get_flag_index(self, target_pos):
        """
            Find the beginning and end of a segment based on the
            change in value of the target_pos array in the dataset,
            as change in value means that the monkey has reached the target
            and a new target is set.

            argument
                target_pos
        """

        # compute difference of target position over the time domain
        target_diff = np.diff(target_pos, axis=1, append=target_pos[:, -1].reshape(2, 1))

        # return indices of starting points of new sessions
        indices = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]

        return indices

    def split_into_segments(self, indices):
        """
            Each segments start & end is defined as:
            [index[i], index[i+1]), [index[i+1], index[i+2]), ...
        """
        # convert data into start and end point of sessions
        start_end = np.array([indices[:-1], indices[1:]])

        return start_end

    def apply_delay(self):
        """
            shift the labels by the delay to account for the biological delay between spikes and movement onset
        """
        # Dimension: No_of_Channels*No_of_Records
        self.samples = self.samples[:, :-self.delay]
        self.labels = self.labels[:, self.delay:]

    def split_data_vanilla(self):
        """
            split data into 4 equal chunks and distribute them into training, testing and validation data
        """
        # This is No. of chunks
        split_num = 4
        total_segments = self.time_segments.shape[1]
        sub_length = int(total_segments / split_num)  # This is no of segments in each chunk
        stride = int(self.stride / DATA_FREQ)
        print(total_segments, sub_length)

        train_len = math.floor(self.train_ratio * sub_length)
        val_len = math.floor((sub_length - train_len) / 2)

        # split the data into 4 equal parts
        # for each part, split the data according to training, testing and validation split
        for split_no in range(split_num):
            for i in range(sub_length):
                # Each segment's Dimension is: No_of_Probes * No_of_Recording
                if i < train_len:
                    self.ind_train += list(np.arange(self.time_segments[0, split_no * sub_length + i],
                                                     self.time_segments[1, split_no * sub_length + i], stride))
                elif train_len <= i < train_len + val_len:
                    self.ind_val += list(np.arange(self.time_segments[0, split_no * sub_length + i],
                                                   self.time_segments[1, split_no * sub_length + i], stride))
                else:
                    self.ind_test += list(np.arange(self.time_segments[0, split_no * sub_length + i],
                                                    self.time_segments[1, split_no * sub_length + i], stride))

    def get_history(self, idx):
        """
            return self.num_steps number of congruent non-overlapping binning windows
        """
        # binning window has a range for "ratio" timesteps
        ratio = int(np.round(self.bin_width / DATA_FREQ))

        # compute indices of congruent binning windows
        mask = idx - np.arange(self.num_steps) * ratio
        return self.samples[:, mask]

    def remove_segments_by_length(self):
        """
            remove sessions that are longer than self.max_segment_length
        """
        self.time_segments = self.time_segments[:, self.time_segments[1, :] - self.time_segments[0, :] <
                                                   self.max_segment_length]

        print("Original Time Segment length is: ", len(self.time_segments))
        print("New Time Segment length is: ", len(self.time_segments))

    def create_dataloader(self, sample, label):
        current_set = self.CustomDataset(sample, label)
        current_loader = DataLoader(
            dataset=current_set,
            batch_size=self.hyperparams['batch_size'],
            drop_last=False,
            shuffle=False)

        return current_loader
