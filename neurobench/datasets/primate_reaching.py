"""
"""

from .dataset import Dataset
import os
import torch
from torch.utils.data import DataLoader
import pickle
import math
import numpy as np
from sklearn.model_selection import KFold
import h5py
from neurobench.preprocessing.primate_reaching import PrimateReachingProcessor
from scipy.signal import convolve2d


class PrimateReaching(Dataset):
    def __init__(self, hyperparams, d_type=torch.float, biological_delay=0, spike_sorting=False, mode="2D",
                 stride=0.004, bin_width=0.208, model_type='ANN', max_segment_len=2000, preprocessor=None):
        super().__init__()

        self.samples = None
        self.labels = None
        self.spike_sorting = spike_sorting
        self.delay = biological_delay
        self.mode = mode
        self.path = hyperparams['dataset_file']
        self.postpr_data_path = hyperparams['postpr_data_path']
        self.stride = stride
        self.bin_width = bin_width
        self.filename = hyperparams['filename']
        self.d_type = d_type
        self.regenerate = hyperparams['regenerate']
        self.num_steps = hyperparams['num_steps']
        self.hyperparams = hyperparams
        self.train_ratio = hyperparams['Train_data_ratio']
        self.bin_process = True
        self.start_end_indices = None
        self.time_segments = None
        self.max_segment_length = 0
        self.model_type = model_type
        self.max_segment_length = max_segment_len
        self.train_index = 0
        self.test_index = 0
        self.validation_index = 0
        self.preprocessor = PrimateReachingProcessor(self.stride, self.bin_width, self.spike_sorting)
        self.segment_no = 0

        #self.load_data()
        self.load_data_paul()

        # Remove segments too long
        self.remove_segments_by_length()

        if self.delay:
            print("Applying Delay to Data")
            self.apply_delay()

        if self.hyperparams["k-fold"]:
            self.split_data_kfold_shuffle()
        else:
            self.split_data_vanilla()

        #if self.mode == "3D":
        #    self.transform_to_3d()

        # print("Post processed sample dimension is: ", self.samples[0].shape)

        # self.create_dataloader()
        
    def __getitem__(self, idx):
        if self.mode == "2D":
            sample = self.samples[:, idx]
            label = self.labels[:, idx]
        elif self.mode == "3D":
            sample = self.samples[:, max(0, idx - int(self.bin_width // 0.004)), idx]
            label = self.labels[:, idx]
        return sample, label

    def __len__(self):
        return self.samples.shape[1]

    def load_data(self):
        # Assume input is the original dataset, instead of the reconstructed one
        if ".mat" in self.filename:
            file_path = os.path.join(self.path, self.filename)
        else:
            file_path = os.path.join(self.path, self.filename + ".mat")
        dataset = h5py.File(file_path, "r")

        spikes = dataset["spikes"][()]  # Get the reference object's locations in the HDF5/mat file
        t = np.squeeze(dataset["t"][()])
        cursor_pos = dataset["cursor_pos"][()]
        target_pos = dataset["target_pos"][()]

        # Define the segments' start & end indices
        self.start_end_indices = np.array(self.get_flag_index(target_pos))
        self.time_segments = np.array(self.split_into_segments(self.start_end_indices))
        print("{} Segments with first one {}-{}".format(len(self.time_segments), self.time_segments[0][0],
                                                        self.time_segments[0][1]))

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

        except:
            self.samples, self.labels = self.preprocessor(spikes, t, cursor_pos, dataset, self.d_type)
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

    def load_data_paul(self):
        # Assume input is the original dataset, instead of the reconstructed one
        if ".mat" in self.filename:
            file_path = os.path.join(self.path, self.filename)
        else:
            file_path = os.path.join(self.path, self.filename + ".mat")
        dataset = h5py.File(file_path, "r")

        spikes = dataset["spikes"][()]  # Get the reference object's locations in the HDF5/mat file
        t = np.squeeze(dataset["t"][()])
        t_diff = t[1]-t[0]
        new_t = np.arange(t[0] - self.bin_width, t[-1], t_diff)
        cursor_pos = dataset["cursor_pos"][()]
        target_pos = dataset["target_pos"][()]

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

        ratio = int(np.round(self.bin_width / t_diff))
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

            :argument
                target_pos
        """

        # OLD CODE
        # target_diff = np.zeros_like(target_pos)
        #
        # target_diff[0, :-1] = np.diff(target_pos[0, :], n=1)
        # target_diff[1, :-1] = np.diff(target_pos[1, :], n=1)
        #
        # x_pos_ind = np.nonzero(target_diff[0, :])[0]
        # y_pos_ind = np.nonzero(target_diff[1, :])[0]
        #
        # index_union1 = np.union1d(x_pos_ind, y_pos_ind)

        # CHANGES
        target_diff = np.diff(target_pos, axis=1, append=target_pos[:, -1].reshape(2, 1))

        index_union = np.nonzero(target_diff.sum(axis=0))[0]

        return index_union

    def split_into_segments(self, indices):
        """
            Each segments start & end is defined as:
            [index[i], index[i+1]), [index[i+1], index[i+2]), ...
        """

        # OLD CODE
        # start_end = []
        # for i in range(len(indices)-1):
        #    start_end.append([indices[i], indices[i+1]])

        # CHANGES
        start_end = np.array([indices[:-1], indices[1:]])

        return start_end

    def apply_delay(self):
        # Dimension: No_of_Channels*No_of_Records
        self.samples = self.samples[:, :-self.delay]
        self.labels = self.labels[:, self.delay:]

    def split_data_vanilla(self):
        # This is No. of chunks
        split_num = 4
        total_segments = self.time_segments.shape[1]
        sub_length = int(total_segments / split_num) # This is no of segments in each chunk
        print(total_segments, sub_length)

        train_len = math.floor((self.train_ratio) * sub_length)
        val_len = math.floor(0.5 * (sub_length - train_len))
        test_len = sub_length - train_len - val_len

        # samples = []
        # labels = []
        for split_no in range(split_num):
            for i in range(sub_length):
                # Each segment's Dimension is: No_of_Probes * No_of_Recording
                if i < train_len:
                    #samples.append(self.samples[:, self.time_segments[0, split_no*sub_length + i]:self.time_segments[1, split_no*sub_length + i]])
                    #labels.append(self.labels[:, self.time_segments[0, split_no*sub_length + i]:self.time_segments[1, split_no*sub_length + i]])
                    self.ind_train += list(np.arange(self.time_segments[0, split_no*sub_length + i], self.time_segments[1, split_no*sub_length + i]))
                elif i >= train_len and i < train_len+val_len:
                    #samples.append(self.samples[:, self.time_segments[0, split_no*sub_length + i]:self.time_segments[1, split_no*sub_length + i]])
                    #labels.append(self.labels[:, self.time_segments[0, split_no*sub_length + i]:self.time_segments[1, split_no*sub_length + i]])
                    self.ind_val += list(np.arange(self.time_segments[0, split_no*sub_length + i], self.time_segments[1, split_no*sub_length + i]))
                else:
                    #samples.append(self.samples[:, self.time_segments[0, split_no*sub_length + i]:self.time_segments[1, split_no*sub_length + i]])
                    #labels.append(self.labels[:, self.time_segments[0, split_no*sub_length + i]:self.time_segments[1, split_no*sub_length + i]])
                    self.ind_test += list(np.arange(self.time_segments[0, split_no*sub_length + i], self.time_segments[1, split_no*sub_length + i]))

        # Dimension is: No_of_segments
        # self.samples = samples  # Each segment's dimension: No_of_Probes * No_of_Recording
        # self.labels = labels
        #print("The final dimension is: ", len(samples), len(labels))

    # def transform_to_3d(self, overlap=True):
    #     # Determine if time window generated overlaps with one another
    #     if not overlap:
    #         advance_num = int(self.stride//0.004)
    #         bin_width_num = advance_num
    #     else:
    #         advance_num = int(self.stride // 0.004)
    #         bin_width_num = int(self.bin_width // 0.004)
    #
    #     new_samples, new_labels = [], []
    #     for sample, label in zip(self.samples, self.labels):
    #         temp_sample = torch.zeros((sample.shape[0], int(sample.shape[1] // advance_num), bin_width_num), dtype=self.d_type)
    #         temp_label = torch.zeros((label.shape[0], int(sample.shape[1] // advance_num)), dtype=self.d_type)
    #
    #         for col in range(temp_sample.shape[1]):
    #             if col < bin_width_num/advance_num:
    #                 bin_start = 0
    #                 bin_end = int(col * advance_num)
    #                 if col == 0:
    #                     bin_end = 1
    #                 temp_sample[:, col, bin_start:bin_end] = sample[:, bin_start: bin_end]
    #             else:
    #                 bin_start = int(col * advance_num - bin_width_num)
    #                 bin_end = int(col * advance_num)
    #                 temp_sample[:, col, :] = sample[:, bin_start: bin_end]
    #
    #             temp_label[:, col] = label[:, col * advance_num]
    #
    #         ## TODO what does this do?
    #         if self.num_steps < bin_width_num:
    #             sum_num = bin_width_num // self.num_steps
    #             temp_sample_num_steps = torch.zeros((temp_sample.shape[0], temp_sample.shape[1], self.num_steps), dtype=self.d_type)
    #             for idx in range(self.num_steps):
    #                 start_idx = idx*sum_num
    #                 end_idx = idx*sum_num + sum_num
    #                 temp_sample_num_steps[:, :, idx] = torch.sum(temp_sample[:, :, start_idx: end_idx], dim=2)
    #             if self.model_type == "ANN":
    #                 new_samples.append(temp_sample_num_steps)
    #             else:
    #                 new_samples.append((temp_sample_num_steps > 0).float())
    #         else:
    #             if self.model_type == "ANN":
    #                 new_samples.append(temp_sample)
    #             else:
    #                 new_samples.append((temp_sample > 0).float())
    #
    #         new_samples[-1] = torch.permute(new_samples[-1], (1, 2, 0))
    #
    #         new_labels.append(temp_label)
    #
    #     print("New Samples Dim", len(new_samples), new_samples[0].size(), new_samples[10].size())
    #     print("New Labels Dim", len(new_labels), new_labels[0].size(), new_labels[10].size())
    #     self.samples = new_samples
    #     self.labels = new_labels

    def split_data_kfold_shuffle(self):
        no_of_segments = len(self.time_segments)

        indices = torch.arange(no_of_segments)

        kf = KFold(n_splits=self.hyperparams["fold_num"], shuffle=True)
        assert kf.get_n_splits(indices) == self.hyperparams["fold_num"], "SKLearn's KFold's n split does not match with hyperparams fold_num"

        for i, (train_index, test_index) in enumerate(kf.split(indices)):
            self.ind_train.append(torch.tensor(train_index))
            split = len(test_index) // 2
            self.ind_val.append(torch.tensor(test_index[:split]))
            self.ind_test.append(torch.tensor(test_index[split:]))

        #total_segments = len(self.time_segments)

        # samples = []
        # labels = []
        # for i in range(total_segments):
        #     # Each segment's Dimension is: No_of_Probes * No_of_Recording
        #     samples.append(self.samples[:, self.time_segments[i][0]:self.time_segments[i][1]])
        #     labels.append(self.labels[:, self.time_segments[i][0]:self.time_segments[i][1]])

        # Dimension is: No_of_segments
        #self.samples = samples  # Each segment's dimension: No_of_Probes * No_of_Recording
        #self.labels = labels
        #print("The final dimension is: ", len(samples), len(labels))

    def remove_segments_by_length(self):
        no_of_segments = len(self.time_segments)
        new_time_segments = []

        # OLD CODE
        #for i in range(no_of_segments):
        #    if (self.time_segments[i][1] - self.time_segments[i][0]) >= self.max_segment_length:
        #        continue
        #    new_time_segments.append(self.time_segments[i])
        #self.time_segments = new_time_segments

        # CHANGES
        self.time_segments = self.time_segments[:, self.time_segments[1, :] - self.time_segments[0, :] <
                                                   self.max_segment_length]

        print("Original Time Segment length is: ", no_of_segments)
        print("New Time Segment length is: ", len(self.time_segments))

    def create_dataloader(self, sample, label):
        current_set = self.CustomDataset(sample, label)
        current_loader = DataLoader(
            dataset=current_set,
            batch_size=self.hyperparams['batch_size'],
            drop_last=False,
            shuffle=False)

        return current_loader