"""
metric_utils.py
------------------------------------------------------------
Utilities required for the metrics.

References
~~~~~~~~~~

Authors
~~~~~~~


"""

from torch.utils.data import Dataset

class NeuroBenchDataset(Dataset):
    """
    Abstract class for NeuroBench datasets. Individual datasets are responsible
    for collating and splitting data. Only testing data is needed right now.
    """
    def __init__(self):
        raise NotImplementedError("Subclasses of NeuroBenchDataset should implement __init__")

    def __len__(self):
        raise NotImplementedError("Subclasses of NeuroBenchDataset should implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses of NeuroBenchDataset should implement __getitem__")