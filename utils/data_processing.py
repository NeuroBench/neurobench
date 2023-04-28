"""
Helper methods for processing our dataset.

@Author: Biyan Zhou
@Date: 28/04/2023
"""

import torch
from torch.utils.data import TensorDataset

def dataset_time_shift(time_shift, data, label):
    """
    There is a time delay about 50ms between spikes from monkeys' brain and detection from probes.
    We need to shift data by a certain time based on the advance time to better match the label.

    :param time_shift: The amount of time shift
    :param data: Data for training
    :param label: Label of data
    :return: Data and label
    """
    if time_shift:
        label = label[:, time_shift:]
        data = data[:, :-time_shift]
    else:
        label = label
        data = data

    return data, label

def data_split(data, label, train_ratio = 0.8, val_ratio = 0.1):
    """
    Randomly split data into training, validation and test set

    :param data: Data for training
    :param label: Label of data
    :param train_ratio: The ratio of training set
    :param val_ratio: The ratio of validation set
    :return: Training, validation and test set
    """
    # check dimension of X,Y
    if data.shape[1] != label.shape[1]:
        diff = abs(data.shape[1] - label.shape[1])
        if label.shape[1] - data.shape[1] < 0:
            data = data[:, : -diff]
        elif label.shape[1] - data.shape[1] > 0:
            label = label[:, : -diff]

    dataset = TensorDataset(data.swapaxes(0,1), label.swapaxes(0,1))

    generator = torch.Generator().manual_seed(1337)
    train_len = int(train_ratio * data.shape[1])
    val_len = int(val_ratio * data.shape[1])
    test_len = int(data.shape[1] - (train_len + val_len))

    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(dataset, [train_len, val_len, test_len], generator=generator)

    return dataset_train, dataset_val, dataset_test

