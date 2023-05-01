import torch
import pickle
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import os

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

def data_processing(file_path, summation, device, advance = 0.1, bin_width = 0.3, postpr_data_path = None, filename = None):
    """
    Processing data from dataset obtained from: https://zenodo.org/record/3854034#.ZEufx-xBwfj.
    Each file in the Dataset contains the following data:
        'chan_names', 'cursor_pos', 'finger_pos', 'spikes', 't', 'target_pos', 'wf'
    For Motor Prediction, we are using the data from 'spikes', 't' and 'cursor_pos'.

    For the .mat file, please make sure it is stored as version v4 (Level 1.0), v6 and v7 to 7.2, as
    Scipy's loadmat() function only support these format.
    It is recommended for the users to use Matlab to convert the .mat files downloaded from Zenodo and
    save the files in a compatible version.

    :param file_path: File path of dataset
    :param summation: Combine every sorted units of every recoding channels or spilt sorted units as different channels
    :param device: Device of data
    :param advance: Frequency of data
    :param bin_width: Time period of each advance time
    :return: Data and label
    """
    full_dataset = loadmat(file_path)
    dataset = full_dataset["a"]

    spikes = dataset["spikes"].item()
    t = dataset["t"].item()
    cursor_pos = dataset["cursor_pos"].item()

    n, u = spikes.shape
    max_len = torch.zeros((1), device=device)

    # Spikes processing
    if summation:
        # Sum units for every channel
        for r in range(n):
            max_lens = torch.zeros((1), device=device)
            for c in range(u):
                max_lens += spikes[:, c][r].shape[0]
            if max_len < max_lens:
                max_len = max_lens

        spike_times = torch.zeros((n, 1, int(max_len.item())), device=device)

        zero_row = torch.zeros((1), device=device)
        for r in range(n):
            tmp = []
            for c in range(u):
                spikes_inner_shape = spikes[r][c].shape[0]
                for s in range(spikes_inner_shape):
                    tmp.append(spikes[r][c][s])
            tmp.sort()
            if len(tmp):
                spike_times[r - int(zero_row.item())][0][0: len(tmp)] = torch.Tensor(np.stack(tmp))[:, 0]

    else:
        # No summation for units
        len_col = torch.zeros((1), device=device)
        for r in range(n):
            for c in range(u):
                max_lens = spikes[r][c].shape[0]
                if max_lens:
                    len_col += 1
                if max_len < max_lens:
                    max_len = max_lens

        spike_times = torch.zeros((int(len_col.item()), 1, max_len), device=device)

        row_record = torch.zeros((1), device=device)
        for r in range(n):
            tmp = []
            for c in range(u):
                spikes_inner_shape = spikes[r][c].shape[0]
                if spikes_inner_shape:
                    spike_times[int(row_record.item())][0][0: spikes_inner_shape] = torch.Tensor(spikes[r][c][:, 0])
                    row_record += 1

    # Extract Y--time steps(Num. of spikes)
    tstart = t[0]
    tend = t[-1]
    time_to_track = np.arange(tstart, tend, advance)
    time_to_track = torch.Tensor(time_to_track)
    time_to_track.to(device)

    t = time_to_track

    testfr = torch.zeros((len(time_to_track), len(spike_times)), device=device)

    for j in tqdm(range(len(spike_times))):
        a = spike_times[j][0]
        for i in range(len(time_to_track)):
            bin_edges = [t[i] - bin_width, t[i]]
            inter = a.gt(bin_edges[0]) & a.le(bin_edges[1])
            testfr[i, j] = torch.sum(inter)

    data = testfr.t()

    # Extract X
    skip_num = round(cursor_pos.shape[0] / data.shape[1])

    X = torch.Tensor(cursor_pos[0:-1:skip_num, :], device=device).t()
    X_tmp = X[:, 1:] - X[:, 0:-1]

    X_start = torch.Tensor(([0], [0]), device=device)
    label = torch.cat((X_start, X_tmp), dim=1)

    # save results
    if filename and postpr_data_path:
        os.makedirs(os.path.join(postpr_data_path, 'input'), exist_ok=True)
        print("Save postprocessed data:", os.path.join(f'{postpr_data_path}', 'input', f'{filename}.pkl'))
        with open(os.path.join(f'{postpr_data_path}', 'input', f'{filename}.pkl'), 'wb') as f:
            pickle.dump(data, f)

        os.makedirs(os.path.join(postpr_data_path, 'label'), exist_ok=True)
        print("Save postprocessed data:", os.path.join(f'{postpr_data_path}', 'label', f'{filename}.pkl'))
        with open(os.path.join(f'{postpr_data_path}', 'label', f'{filename}.pkl'), 'wb') as f:
            pickle.dump(label, f)

    return data, label
