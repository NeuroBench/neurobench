"""
"""
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self):
        self.samples = None
        self.labels = None
        self.ind_train, self.ind_val, self.ind_test = [], [], []

    def get_feature_size(self):
        return self.samples.size()

    def get_labels_size(self):
        return self.labels.size()

    def __getitem__(self, idx):
        sample = self.samples[idx, :, :]
        label = self.labels[idx, :, :]
        return sample, label

class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, idx):
        sample = self.samples[:, idx, :]
        label = self.labels[:, idx]
        return sample, label

    def __len__(self):
        return self.samples.shape[1]