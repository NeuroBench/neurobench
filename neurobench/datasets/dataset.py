"""
=====================================================================
Project:      NeuroBench
File:         dataset.py
Description:  Python code describing virtual class for Dataset
Date:         11. May 2023
=====================================================================
Copyright stuff
=====================================================================
"""
from torch.utils.data import Dataset


class Dataloader(Dataset):
    """
    Virtual class for Dataloaders

    Methods
    ----------
    __len__
        overload len method and returns length of dataset
    __getitem__
        overload getter method
    evaluate
        runs testing / validation pipeline
    """
    def __init__(self):
        self.samples = None
        self.labels = None
        self.ind_train, self.ind_val, self.ind_test = None, None, None

    def __len__(self):
        """
        returns number of samples in dataset

        Returns
        ----------
        length:  int
            number of samples in dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Getter method for Dataloader
        Can be overwritten

        Returns
        ----------
        idx : int
            index of sample. Can be used for generator

        Returns
        ----------
        sample:  ndarray
            individual data sample (samples x channels x window)
        labels:  ndarray
            corresponding finger position (samples x features x window)
        """
        sample = self.samples[idx, :, :]
        label = self.labels[idx, :, :]

        return sample, label
