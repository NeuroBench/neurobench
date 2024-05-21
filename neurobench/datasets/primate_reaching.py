"""
This file contains code from PyTorch Vision (https://github.com/pytorch/vision) which is licensed under BSD 3-Clause License.
These snippets are the Copyright (c) of Soumith Chintala 2016. All other code is the Copyright (c) of the NeuroBench Developers 2023.
"""

from .dataset import NeuroBenchDataset
from .utils import check_integrity, download_url
from torch.utils.data import Dataset
import os
import torch
import math
import numpy as np
import h5py
from scipy.signal import convolve2d
from urllib.error import URLError

# The spikes recorded in the Primate Reaching datasets have an interval of 4ms.
SAMPLING_RATE = 4e-3


class PrimateReaching(NeuroBenchDataset):
    """
    Dataset for the Primate Reaching Task.

    The Dataset can be downloaded from the following website:
    https://zenodo.org/record/583331

    For this task, the following files are selected:
    1. indy_20170131_02.mat
    2. indy_20160630_01.mat
    3. indy_20160622_01.mat
    4. loco_20170301_05.mat
    5. loco_20170215_02.mat
    6. loco_20170210_03.mat

    The description of the structure of the dataset can be found on the website
    in the section: Variable names.

    Once these .mat files are downloaded, store them in the same directory.

    """

    url = "https://zenodo.org/record/583331/files/"

    md5s = {
        "indy_20170131_02.mat": "2790b1c869564afaa7772dbf9e42d784",
        "indy_20160630_01.mat": "197413a5339630ea926cbd22b8b43338",
        "indy_20160622_01.mat": "c33d5fff31320d709d23fe445561fb6e",
        "loco_20170301_05.mat": "47342da09f9c950050c9213c3df38ea3",
        "loco_20170215_02.mat": "739b70762d838f3a1f358733c426bb02",
        "loco_20170210_03.mat": "4cae63b58c4cb9c8abd44929216c703b",
    }

    def __init__(
        self,
        file_path,
        filename,
        num_steps,
        train_ratio=0.8,
        label_series=False,
        biological_delay=0,
        spike_sorting=False,
        stride=0.004,
        bin_width=0.028,
        max_segment_length=2000,
        split_num=1,
        remove_segments_inactive=False,
        download=True,
    ):
        """
        Initialises the Dataset for the Primate Reaching Task.

        Args:
            file_path (str): The path to the directory storing the matlab files.
            filename (str): The name of the file that will be loaded.
            num_steps (int): Number of consecutive timesteps that are included per sample.
                             In the real-time case, this should be 1.
            train_ratio (float): ratio for how the dataset will be split into training/(val+test) set.
                                 Default is 0.8 (80% of data is training).
            label_series (bool): Whether the labels are series or not. Useful for training with multiple
                                 timesteps. Default is False.
            biological_delay (int): How many steps of delay is to be applied to the dataset. Default is 0
                                    i.e. no delay applied.
            spike_sorting (bool): Apply spike sorting for processing raw spike data. Default is False.
            stride (float):  How many steps are taken when moving the bin_window. Default is 0.004 (4ms).
            bin_width (float): The size of the bin_window. Default is 0.028 (28ms).
            max_segment_length: Define the upper limits of a segment. Default is 2000 data points (8s)
            split_num (int): The number of chunks to break the timeseries into. Default is 1 (no splits).
            remove_segments_inactive (bool): Whether to remove segments longer than max_segment_length,
                                             which represent subject inactivity. Default is False.
            download (bool): If True, downloads the dataset from the internet and puts it in root
                             directory. If dataset is already downloaded, it will not be downloaded again.

        """
        self.url = "https://zenodo.org/record/583331/files/"

        self.md5s = {
            "indy_20170131_02.mat": "2790b1c869564afaa7772dbf9e42d784",
            "indy_20160630_01.mat": "197413a5339630ea926cbd22b8b43338",
            "indy_20160622_01.mat": "c33d5fff31320d709d23fe445561fb6e",
            "loco_20170301_05.mat": "47342da09f9c950050c9213c3df38ea3",
            "loco_20170215_02.mat": "739b70762d838f3a1f358733c426bb02",
            "loco_20170210_03.mat": "4cae63b58c4cb9c8abd44929216c703b",
        }

        # The samples and labels of the dataset
        self.samples = None
        self.labels = None

        # used for input data file management
        self.filename = filename if filename[-4:] == ".mat" else filename + ".mat"
        self.file_path = os.path.join(file_path, self.filename)

        if download:
            self.download()

        # test filepath
        assert os.path.exists(self.file_path)

        # related to processing of spike data
        self.spike_sorting = spike_sorting
        self.delay = biological_delay
        self.stride = stride
        self.bin_width = bin_width
        self.num_steps = num_steps
        self.train_ratio = train_ratio
        self.label_series = label_series
        self.ratio = int(np.round(self.bin_width / SAMPLING_RATE))

        # test parameters
        assert self.delay >= 0
        assert self.stride >= SAMPLING_RATE
        assert (
            self.bin_width >= SAMPLING_RATE
        ), "The binning window has to be greater than the sampling size (i.e. 0.004s)"
        assert self.num_steps >= 1
        assert 0 <= self.train_ratio <= 1

        # Defines the beginning and end of each segment.
        self.start_end_indices = None
        self.time_segments = None

        # Defines the maximum length of a segment.
        self.max_segment_length = max_segment_length
        assert self.max_segment_length >= 0

        self.split_num = split_num

        # These lists store the index of segments that belongs to training/validation/test set
        self.ind_train, self.ind_val, self.ind_test = [], [], []

        if "indy" in filename:
            self.input_feature_size = 96
        elif "loco" in filename:
            self.input_feature_size = 192
        else:
            raise ValueError(
                "Unexpected filename. Filename should be of either indy or loco"
            )

        self.load_data()

        if self.delay > 0:
            self.apply_delay()

        if remove_segments_inactive and self.max_segment_length > 0:
            self.valid_segments = self.remove_segments_by_length()
        else:
            self.valid_segments = np.arange(self.time_segments.shape[0])

        self.split_data()

    def __len__(self):
        return len(self.ind_train) + len(self.ind_test) + len(self.ind_val)

    def __getitem__(self, idx):
        """Getter method of the dataloader."""
        # compute indices of congruent binning windows
        mask = idx - np.arange(self.num_steps) * self.ratio
        if self.label_series:
            samples = self.samples[:, mask].transpose(0, 1)
            labels = self.labels[:, mask].transpose(0, 1)
            return samples, labels
        else:
            return self.samples[:, mask].transpose(0, 1), self.labels[:, idx]

    def _check_exists(self, file_path, md5) -> bool:
        return check_integrity(file_path, md5)

    def download(self):
        """Download the Primate Reaching data if it doesn't exist already."""
        md5 = self.md5s[self.filename]

        if self._check_exists(self.file_path, md5):
            return

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # download file
        url = f"{self.url}{self.filename}"
        try:
            print(f"Downloading {url}")
            download_url(url, self.file_path, md5=md5)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print()

    def load_data(self):
        """Load the data from the matlab file and spike data if spike data has been
        processed and stored already."""
        # Assume input is the original dataset, instead of the reconstructed one
        print(f"Loading {self.filename}")
        dataset = h5py.File(self.file_path, "r")

        # extract data from datafile
        spikes = dataset["spikes"][
            ()
        ]  # Get the reference object's locations in the HDF5/mat file
        cursor_pos = dataset["cursor_pos"][()]
        target_pos = dataset["target_pos"][()]
        t = np.squeeze(dataset["t"][()])
        new_t = np.arange(t[0] - self.bin_width, t[-1], SAMPLING_RATE)

        # Define the segments' start & end indices
        self.start_end_indices = np.array(self.get_flag_index(target_pos))
        self.time_segments = np.array(
            self.split_into_segments(self.start_end_indices, target_pos.shape[1])
        )

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
        if self.ratio != 1:
            binned_spike_train = convolve2d(
                spike_train, np.ones((1, self.ratio)), mode="valid"
            )
        else:
            binned_spike_train = spike_train

        # Dimensions: (channels x timesteps)
        self.samples = torch.from_numpy(binned_spike_train).float()
        # Dimensions: (nr_features x timesteps)
        self.labels = torch.from_numpy(cursor_pos).float()

        # convert position to velocity
        self.labels = torch.gradient(self.labels, dim=1)[0]

    def apply_delay(self):
        """Shift the labels by the delay to account for the biological delay between
        spikes and movement onset."""
        # Dimension: No_of_Channels*No_of_Records
        self.samples = self.samples[:, : -self.delay]
        self.labels = self.labels[:, self.delay :]

    def split_data(self):
        """Split segments into training/validation/test set."""
        # This is No. of chunks
        split_num = self.split_num
        total_segments = self.time_segments.shape[0]
        sub_length = int(
            total_segments / split_num
        )  # This is no of segments in each chunk
        stride = int(self.stride / SAMPLING_RATE)
        # print(total_segments, sub_length)

        train_len = math.floor(self.train_ratio * sub_length)
        val_len = math.floor((sub_length - train_len) / 2)

        # offset = int(np.round(self.bin_width / SAMPLING_RATE)) * self.num_steps
        offset = 0

        # split the data into 4 equal parts
        # for each part, split the data according to training, testing and validation split
        for split_no in range(split_num):
            for i in range(sub_length):
                # Each segment's Dimension is: No_of_Probes * No_of_Recording
                if i < train_len and i in self.valid_segments:
                    self.ind_train += list(
                        np.arange(
                            offset + self.time_segments[split_no * sub_length + i, 0],
                            self.time_segments[split_no * sub_length + i, 1],
                            stride,
                        )
                    )
                elif train_len <= i < train_len + val_len and i in self.valid_segments:
                    self.ind_val += list(
                        np.arange(
                            offset + self.time_segments[split_no * sub_length + i, 0],
                            self.time_segments[split_no * sub_length + i, 1],
                            stride,
                        )
                    )
                elif i in self.valid_segments:
                    self.ind_test += list(
                        np.arange(
                            offset + self.time_segments[split_no * sub_length + i, 0],
                            self.time_segments[split_no * sub_length + i, 1],
                            stride,
                        )
                    )

    def remove_segments_by_length(self):
        """Remove the segments where its duration exceeds the limit set by
        max_segment_length."""
        return np.nonzero(
            self.time_segments[:, 1] - self.time_segments[:, 0]
            < self.max_segment_length
        )[0]

    @staticmethod
    def split_into_segments(indices, last_idx):
        """Combine the start and end index into a NumPy array."""
        indices = np.insert(indices, 0, 0)
        indices = np.append(indices, [last_idx])
        start_end = np.array([indices[:-1], indices[1:]])

        return np.transpose(start_end)

    @staticmethod
    def get_flag_index(target_pos):
        """Find where each segment begins and ends."""
        target_diff = np.diff(
            target_pos, axis=1, append=target_pos[:, -1].reshape(2, 1)
        )

        indices = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]

        return indices
