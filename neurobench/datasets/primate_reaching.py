"""
=====================================================================
Project:      NeuroBench
File:         primate_reaching.py
Description:  Python code describing dataloader for the motor prediction task
Date:         11. May 2023
=====================================================================
Copyright stuff
=====================================================================
"""

from neurobench.datasets.dataset import Dataloader
from neurobench.datasets.utils import select_file, valid_path
from scipy.io import loadmat
import h5py
import numpy as np
import torch


class PrimateReaching(Dataloader):
    """
    Dataloader for the Primate Reaching Task

    Parameters
    ----------
    path    :  str
        path to MATLAB datafile
    biological_delay     :  int
        biological delay between spike train and movement onset. Labels are shifted by this amount
    window  : int
        window that is extracted from the data
    stride  : int
        stride of the sliding window
    use_spike_sorting   : bool
        if true, use spike sorted data, otherwise combine units into channels (summation vs non-summation)
    splits  : list
        data split in ms for training, validation and testing data respectively

    Methods
    ----------
    __getitem__
        overload getter function, gets sample and corresponding label from dataset
    load_data
        Load data from MATLAB file
    apply_delay
        apply a specific delay for the labels
    split_data
        split data into training, testing and validation data
    """
    def __init__(self, path=None, biological_delay=140, window=2500, stride=10, use_spike_sorting=False,
                 splits=(10000, 5000, 5000)):

        super().__init__()

        # parameters
        # data (timesteps x channels)
        self.samples = None
        # data (timesteps x nr_features)
        self.labels = None

        self.window = window
        self.stride = stride
        self.use_spike_sorting = use_spike_sorting
        self.delay = biological_delay
        self.splits = splits

        # if no file selected, open dialog window to select data file
        if path is None:
            path = select_file()

        # ensure file exists and is valid .mat file
        if valid_path(path):
            self.path = path

        # load data
        self.load_data()

        # shift labels by a set delay
        self.apply_delay()

        # split data into training, testing and validation data
        self.split_data()

    def __getitem__(self, idx):
        """
        Getter method for dataloader

        Parameters
        ----------
        idx : int
            index of sample. indices are assigned in split_data() method

        Returns
        ----------
        sample  : ndarrax
            spike train                    (channels x self.window)
        labels  : ndarray
            corresponding finger position  (2 x self.window)
        """
        sample = self.samples[:, idx:idx + self.window]
        labels = self.labels[:, idx:idx + self.window]

        return sample, labels

    def load_data(self):
        """
        Load data from MATLAB file
        Matlab v7.3 cannot be opened with scipy loadmat.

        """
        try:
            df = loadmat(self.path)
            t = df['a']['t'].item().transpose()
            labels = df['a']['cursor_pos'].item().transpose()
            spikes = df['a']['spikes'].item().transpose()

        except NotImplementedError:
            # open Matlab v7.3 / HDF5 file
            df = h5py.File(self.path, 'r')

            # extract data
            t = df['t'][()]
            labels = df['cursor_pos'][()]
            spikes = df['spikes'][()]

        # extract timestep and allocate memory for spike_train
        spike_train = np.zeros((*spikes.shape, t.shape[1]), dtype=np.int8)

        # iterate over hdf5 dataframe and preprocess data
        for row_idx, row in enumerate(spikes):
            for col_idx, element in enumerate(row):
                # get indices of spikes and convert data to spike train
                bins, _ = np.histogram(df[element][()], bins=t.squeeze())
                idx = np.nonzero(bins)[0] + 1
                spike_train[row_idx, col_idx, idx] = 1

        if self.use_spike_sorting:
            # if using spike sorting, reshape # channels x # units into a single dimension => # features
            spike_train = np.transpose(spike_train, (2, 1, 0)).reshape(t.shape[1], -1)

            # remove empty channels
            spike_train = spike_train[:, spike_train.any(axis=0)]
            spike_train = spike_train.transpose()
        else:
            # combine units into channels
            spike_train = np.bitwise_or.reduce(spike_train, axis=0)

        # Dimensions: (channels x timesteps)
        self.samples = torch.from_numpy(spike_train).float()
        # Dimensions: (nr_features x channels)
        self.labels = torch.from_numpy(labels).float()

        # convert position to velocity
        self.labels = torch.gradient(self.labels, dim=1)[0]

    def apply_delay(self):
        """
        Shift labels by amount specified in delay to account for biological delay of movements

        """
        if self.delay:
            self.samples = self.samples[:, :-self.delay]
            self.labels = self.labels[:, self.delay:]

    def split_data(self):
        """
        Split data into training, validation and testing data according to the timesteps specified in self.splits
        Data is split into chunks of sum(self.splits) ms, chunks are then split into training, validation and
        testing data

        """
        # split indices into sum(self.splits) chunks
        length = self.samples.shape[1]
        cutoff = length % sum(self.splits)
        indices = torch.arange(length - cutoff).reshape(-1, sum(self.splits))

        # extract indices based on window length and stride without overlap between datasets
        self.ind_train = indices[:, :self.splits[0] - self.window + 1:self.stride].flatten()
        self.ind_val = indices[:, self.splits[0]:-self.splits[2] - self.window + 1:self.stride].flatten()
        self.ind_test = indices[:, -self.splits[2]:- self.window + 1:self.stride].flatten()