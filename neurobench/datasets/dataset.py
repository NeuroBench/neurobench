from torch.utils.data import Dataset
from abc import abstractmethod


class NeuroBenchDataset(Dataset):
    """
    Abstract class for NeuroBench datasets.

    Individual datasets are responsible for collating and splitting data. The top-level
    benchmark expects that the dataset is wrapped into a DataLoader.

    """

    def __init__(self):
        """Init dataset."""
        super().__init__()

    @abstractmethod
    def __len__(self):
        """Returns length of dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Returns a sample from the dataset."""
        pass
