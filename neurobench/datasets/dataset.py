from torch.utils.data import Dataset


class NeuroBenchDataset(Dataset):
    """
    Abstract class for NeuroBench datasets.

    Individual datasets are responsible for collating and splitting data. The top-level
    benchmark expects that the dataset is wrapped into a DataLoader.

    """

    def __init__(self):
        """Init dataset."""
        raise NotImplementedError(
            "Subclasses of NeuroBenchDataset should implement __init__"
        )

    def __len__(self):
        """Returns length of dataset."""
        raise NotImplementedError(
            "Subclasses of NeuroBenchDataset should implement __len__"
        )

    def __getitem__(self, idx):
        """Returns a sample from the dataset."""
        raise NotImplementedError(
            "Subclasses of NeuroBenchDataset should implement __getitem__"
        )
