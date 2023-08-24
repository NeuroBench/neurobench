from .dataset import NeuroBenchDataset
from torch.utils.data import Dataset
import os
# from neurobench.preprocessing import PreProcessor
from neurobench.preprocessing.primate_reaching import PrimateReachingProcessor
import torch
from torch.utils.data import DataLoader
import pickle
import math
import numpy as np
import h5py
from tqdm import tqdm

# The spikes recorded in the Primate Reaching datasets have an interval of 4ms.
SAMPLING_RATE = 0.004

class PrimateReaching(NeuroBenchDataset):
    """
        Dataset for the Primate Reaching Task
    """
    def __init__(self, file_path, postpr_data_path, filename, first_process, num_steps, train_ratio=0.8,
                 mode="3D", model_type="ANN", dtype=torch.float, biological_delay=0,
                 spike_sorting=False, stride=0.004, bin_width=0.200, max_segment_length=2000):
        """
            Initialises the Dataset for the Primate Reaching Task.

            Args:
                file_path (str): The path to the directory storing the matlab files.
                postpr_data_path (str): The path that stores the processed spike data, which can be loaded
                                        once they're generated.
                filename (str): The name of the file that will be loaded.
                first_process (bool): check if it's the first time running this class. If True, this class
                                      will process through the spike data which can be stored in the path
                                      provided with postpr_data_path
                num_steps (int): number of timesteps the data will be split into.
                train_ratio (float): ratio for how the dataset will be split into training/(val+test) set.
                                     Default is 0.8 (80% of data is training).
                mode (str): The data processed will be either "2D" (data_points, input_features) or 
                            "3D" (data_points, num_steps, input_features). Default is "3D".
                model_type (str): The type of model that will be using the dataset. Currently expects "ANN",
                                  "SNN" or "LSTM" (LSTM to be added later). Default is "ANN
                dtype: PyTorch tensor's data type. Default is torch.float.
                biological_delay (int): How many steps of delay is to be applied to the dataset. Default is 0
                                        i.e. no delay applied.
                spike_sorting (bool): Apply spike sorting for processing raw spike data. Default is False.
                stride (float):  How many steps are taken when moving the bin_window. Default is 0.004 (4ms).
                bin_width (float): The size of the bin_window. Default is 0.2 (200ms).
                max_segment_len: Define the upper limits of a segment. Default is 2000 data points (8s)
        """
        # The samples and labels of the dataset
        self.samples = None
        self.labels = None
        # used for input data file management
        self.path = file_path
        self.postpr_data_path = postpr_data_path
        self.filename = filename
        self.first_process = first_process
        # related to processing of spike data
        self.spike_sorting = spike_sorting
        self.delay = biological_delay
        self.stride = stride
        self.bin_width = bin_width
        self.num_steps = num_steps
        self.train_ratio = train_ratio
        # Defines the beginning and end of each segments.
        self.start_end_indices = None
        self.time_segments = None
        # Defines the maximum length of a segment.
        self.max_segment_length = max_segment_length
        # Dataset use mode
        self.mode = mode
        self.model_type = model_type
        # These lists store the index of segments that belongs to training/validation/test set
        self.ind_train, self.ind_val, self.ind_test = [], [], []
        self.segment_no = 0
        self.dtype = dtype

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

        if mode == "3D":
            self.bin_processing_3d()

    def __len__(self):
        return len(self.samples[self.segment_no])
    
    def __getitem__(self, idx):
        if self.mode == "2D":
            sample = self.samples[self.segment_no][:, idx]
            label = self.labels[self.segment_no][:, idx]
        elif self.mode == "3D":
            sample = self.samples[self.segment_no][idx, :, :]
            label = self.labels[self.segment_no][idx, :]
        return sample, label

    def load_data(self):
        """
            Load the data from the matlab file and spike data 
            if spike data has been processed and stored already
        """
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
        self.time_segments = self.split_into_segments(self.start_end_indices)

        try:
            if self.first_process:
                raise Exception("first time processing data...")

            with open(os.path.join(f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl'), 'rb') as f:
                self.samples = pickle.load(f)

            with open(os.path.join(f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl'), 'rb') as f:
                self.labels = pickle.load(f)

            print("Sample shape: {}, Label shape: {}".format(self.samples.shape, self.labels.shape))

        except:
            self.samples, self.labels = first_time_sample_label_split(spikes, t, cursor_pos, 
                                                                      dataset, self.dtype, step=SAMPLING_RATE, bin_width_time=SAMPLING_RATE)
            print("Sample shape: {}, Label shape: {}".format(self.samples.shape, self.labels.shape))

            if self.filename and self.postpr_data_path:
                os.makedirs(os.path.join(self.postpr_data_path, 'input'), exist_ok=True)
                with open(os.path.join(f'{self.postpr_data_path}', 'input', f'{self.filename}.pkl'), 'wb') as f:
                    pickle.dump(self.samples, f)

                os.makedirs(os.path.join(self.postpr_data_path, 'label'), exist_ok=True)
                with open(os.path.join(f'{self.postpr_data_path}', 'label', f'{self.filename}.pkl'), 'wb') as f:
                    pickle.dump(self.labels, f)

    def get_flag_index(self, target_pos):
        """
            Find where each segment begins and ends
        """
        target_diff = np.diff(target_pos, axis=1, append=target_pos[:, -1].reshape(2, 1))

        index_union = np.nonzero(np.sum(np.abs(target_diff), axis=0))[0]

        return index_union
    
    def split_into_segments(self, indices):
        """
            Combine the start and end index into a NumPy array.
        """
        start_end = np.array([indices[:-1], indices[1:]])

        return np.transpose(start_end)
    
    def apply_delay(self):
        """
            
        """
        # Dimension: No_of_Channels*No_of_Records
        self.samples = self.samples[:, :-self.delay]
        self.labels = self.labels[:, self.delay:]

    def split_data(self):
        """
            Split segments into training/validation/test set
        """
        split_num = 4
        total_segments = len(self.time_segments)
        sub_length = int(total_segments / split_num) # This is no of segments in each chunk
        train_len = math.floor((self.train_ratio) * sub_length)
        val_len = math.floor(0.5 * (sub_length - train_len))
        test_len = sub_length - train_len - val_len

        samples, labels = [], []
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

        self.samples, self.labels = samples, labels

    def bin_processing_3d(self, overlap=True):
        """
            process each segment so that each datapoint contains spike data over a time bin window,
            where the time bin window can then be sub-divided into num_steps time steps.
            If model_type is "ANN", the data contains integers.
            If model_type is "SNN", the data is either 0 or 1.
        """
        # Determine if time window generated overlaps with one another
        if not overlap:
            advance_num = int(self.stride//0.004)
            bin_width_num = advance_num
        else:
            advance_num = int(self.stride//0.004)
            bin_width_num = int(self.bin_width//0.004)
        
        new_samples, new_labels = [], []
        for sample, label in zip(self.samples, self.labels):
            temp_sample = torch.zeros((sample.shape[0], int(sample.shape[1] // advance_num), bin_width_num), dtype=self.dtype)
            temp_label = torch.zeros((label.shape[0], int(sample.shape[1] // advance_num)), dtype=self.dtype)

            for col in range(temp_sample.shape[1]):
                if col <  bin_width_num/advance_num:
                    bin_start = 0
                    bin_end = int(col * advance_num)
                    if col == 0:
                        bin_end = 1
                    temp_sample[:, col, bin_start:bin_end] = sample[:, bin_start: bin_end]
                    # continue
                else:
                    bin_start = int(col * advance_num - bin_width_num)
                    bin_end = int(col * advance_num)
                    temp_sample[:, col, :] = sample[:, bin_start: bin_end]

                temp_label[:, col] = label[:, col * advance_num]

            if self.num_steps < bin_width_num:
                sum_num = bin_width_num // self.num_steps
                temp_sample_num_steps = torch.zeros((temp_sample.shape[0], temp_sample.shape[1], self.num_steps), dtype=self.dtype)
                for idx in range(self.num_steps):
                    start_idx = idx*sum_num
                    end_idx = idx*sum_num + sum_num
                    temp_sample_num_steps[:, :, idx] = torch.sum(temp_sample[:, :, start_idx: end_idx], dim=2)

                if self.model_type != 'SNN':
                    new_samples.append(temp_sample_num_steps)
                else:
                    new_samples.append((temp_sample_num_steps > 0).float())
            else:

                if self.model_type != 'SNN':
                    new_samples.append(temp_sample)
                else:
                    new_samples.append((temp_sample > 0).float())

            new_samples[-1] = torch.permute(new_samples[-1], (1, 2, 0))
            new_labels.append(temp_label)

        self.samples = new_samples
        self.labels = new_labels

    def remove_segments_by_length(self):
        """
            remove the segments where its duration exceeds the limit set by
            max_segment_length
        """
        self.time_segments = self.time_segments[self.time_segments[:, 1] - self.time_segments[:, 0] <
                                                self.max_segment_length, :]
        
    def create_dataloader(self, split_type: str, batch_size=256, shuffle=True, drop_last=False):
        """
            Helper method for creating a PyTorch DataLoader based on the split_type.
            Args:
                split_type (str): Defines the split type that will be loaded into the DataLoader.
                                  Can be of the type "Train", "Validation" or "Test".
        """
        if split_type == "Train":
            indices = self.ind_train
        elif split_type == "Validation":
            indices = self.ind_val
        elif split_type == "Test":
            indices = self.ind_test
        else:
            raise ValueError("Unexpected split type. Expects Train, Validation or Test")
        
        sample_set, label_set = [], []
        for idx in indices:
            sample_set.append(self.samples[idx])
            label_set.append(self.labels[idx])

        sample_set = torch.cat(sample_set, dim=0)
        label_set = torch.cat(label_set, dim=1)

        class CustomDataset(Dataset):
            def __init__(self, samples, labels):
                self.samples = samples
                self.labels = labels

            def __getitem__(self, idx):
                sample = self.samples[idx, :, :]
                label = self.labels[:, idx]
                return sample, label

            def __len__(self):
                return self.samples.shape[0]
            
        data_set = CustomDataset(sample_set, label_set)
        data_loader = DataLoader(dataset=data_set, batch_size=batch_size,
                                 drop_last=drop_last, shuffle=shuffle)
        
        return data_loader


def first_time_sample_label_split(spikes, t, cursor_pos, dataset, dtype, 
                                  step=0.004, bin_width_time=0.004, spike_sorting=False):
    """
        Helper method to help process the spike data the first time.
        The method will iterate through the spikes array stored in the Matlab file
        and process it so that each probe's data is cleaned up.

        The label is also generated with this function, where the cursor_pos array is
        converted into a velocity array that's used as the label for this task.
    """
    no_of_units, no_of_probes = spikes.shape # Dimension is 5*96
    max_len = torch.zeros((1))

    tstart = t[0]
    tend = t[-1]

    # Spikes processing
    if not spike_sorting:
        for r in range(no_of_probes):
            max_lens = torch.zeros((1))
            for c in range(no_of_units):
                ref = spikes[c, r]
                actual_obj = dataset[ref]
                if actual_obj.shape[-1] == 2:
                    continue
                max_lens += actual_obj.shape[-1]
            if max_len < max_lens:
                max_len = max_lens

        spike_times = torch.zeros((no_of_probes, 1, int(max_len.item())))

        zero_row = torch.zeros((1))
        for r in range(no_of_probes):
            tmp = []
            for c in range(no_of_units):
                spikes_inner_shape = dataset[spikes[c, r]].shape[-1]
                if spikes_inner_shape == 2:
                    continue
                local_spikes = dataset[spikes[c, r]]
                for s in range(spikes_inner_shape):
                    tmp.append(local_spikes[:, s])
            tmp.sort()
            if len(tmp):
                spike_times[r - int(zero_row.item())][0][0: len(tmp)] = torch.Tensor(np.stack(tmp))[:, 0]

    else:
        # No spike_sorting for units
        len_col = torch.zeros((1))
        for r in range(no_of_probes):
            for c in range(no_of_units):
                # max_lens = spikes[r][c].shape[0]
                max_lens = dataset[spikes[c, r]].shape[-1]
                if max_lens <= 2 and np.max(dataset[spikes[c, r]][:]) < tstart:
                    continue
                if max_lens:
                    len_col += 1
                if max_len < max_lens:
                    max_len = max_lens
                
        spike_times = torch.zeros((int(len_col.item()), 1, max_len))

        row_record = torch.zeros((1))
        for r in range(no_of_probes):
            tmp = []
            for c in range(no_of_units):
                spikes_inner_shape = dataset[spikes[c, r]].shape[-1]
                if spikes_inner_shape != 2:
                    local_spikes = dataset[spikes[c, r]]
                    spike_times[int(row_record.item())][0][0: spikes_inner_shape] = torch.Tensor(local_spikes[0, :])
                    row_record += 1

    # Extract data--time steps(Num. of spikes)
    time_to_track = np.arange(tstart, tend+step, step)
    time_to_track = torch.Tensor(time_to_track)

    testfr = torch.zeros((len(time_to_track), len(spike_times)))

    for j in tqdm(range(len(spike_times))):
        a = spike_times[j][0]
        for i in range(len(time_to_track)):
            bin_edges = [time_to_track[i] - bin_width_time, time_to_track[i]]
            inter = a.gt(bin_edges[0]) & a.le(bin_edges[1])
            testfr[i, j] = torch.sum(inter)

    data = torch.swapaxes(testfr, 0, 1)

    ## Extract label
    skip_num = round(cursor_pos.shape[-1] / data.shape[1])

    cursor = torch.tensor(cursor_pos[:, ::skip_num], dtype=dtype)
    cursor_tmp = cursor[:, 1:] - cursor[:, 0:-1]
    cursor_start = torch.Tensor(([0], [0]))
    label = torch.cat((cursor_start, cursor_tmp), dim=1)

    return (data, label)