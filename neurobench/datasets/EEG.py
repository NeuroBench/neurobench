from neurobench.datasets.dataset import Dataset
import numpy as np
import torch
import os
from .utils import download_url
from urllib.error import URLError

"""
Preprocessed EEG Motor Imagery (MI) dataset derived from the Lee2019 dataset
(Lee et al., 2019, "EEG dataset and OpenBMI toolbox for three BCI paradigms:
An investigation into BCI illiteracy"), adapted for the THOR challenge.

The original Lee2019 dataset contains EEG recordings from 54 subjects performing
left-hand and right-hand motor imagery tasks. This version provides preprocessed
train/validation splits ready for use in the NeuroBench THOR benchmark.

Data is automatically downloaded from:
https://huggingface.co/datasets/NeuroBench/thor_eeg_mi

Original dataset reference:
    Lee, M.-H., et al. (2019). EEG dataset and OpenBMI toolbox for three BCI
    paradigms: An investigation into BCI illiteracy.
    GigaScience, 8(5), giz002. https://doi.org/10.1093/gigascience/giz002
"""

BASE_URL = "https://huggingface.co/datasets/NeuroBench/thor_eeg_mi/resolve/main"

FILES = {
    "train_X": "train_X.npy",
    "train_y": "train_y.npy",
    "val_X": "val_X.npy",
    "val_y": "val_y.npy",
}


class ThorEEGMI(Dataset):
    """
    Preprocessed Lee2019 EEG Motor Imagery dataset adapted for the THOR challenge.

    This dataset is derived from the Lee2019 benchmark (Lee et al., 2019), which
    recorded EEG from 54 subjects performing left-hand and right-hand motor imagery
    tasks using a 62-channel cap at 1000 Hz. The data here is a preprocessed and
    re-split version tailored for the NeuroBench THOR challenge, provided as
    train/validation splits of trials with shape ``(n_trials, n_timesteps, n_channels)``.

    Each sample is a single motor imagery trial and the target is an integer class
    label (0 = right hand, 1 = left hand).

    Args:
        root (str): Root directory where the dataset files are stored (or
            will be downloaded to).
        split (str): Which split to load. One of ``"train"`` or ``"val"``.
        download (bool): If ``True``, downloads the dataset files from
            HuggingFace if they are not already present in ``root``.

    Reference:
        Lee, M.-H., et al. (2019). EEG dataset and OpenBMI toolbox for three BCI
        paradigms: An investigation into BCI illiteracy.
        GigaScience, 8(5), giz002. https://doi.org/10.1093/gigascience/giz002

    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
    ):
        super().__init__()

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.root = root
        self.split = split

        os.makedirs(self.root, exist_ok=True)

        if download:
            self._download()

        self._load_data()

    def _file_path(self, key: str) -> str:
        """Return the full local path for a dataset file key."""
        return os.path.join(self.root, FILES[key])

    def _download(self):
        """Download all four dataset files if not already present."""
        for key, filename in FILES.items():
            dest = self._file_path(key)
            if os.path.exists(dest):
                continue
            url = f"{BASE_URL}/{filename}"
            try:
                print(f"Downloading {url}")
                download_url(url, dest)
            except URLError as error:
                raise RuntimeError(
                    f"Failed to download {filename}:\n{error}"
                ) from error

    def _load_data(self):
        """Load the appropriate split into tensors."""
        x_key = f"{self.split}_X"
        y_key = f"{self.split}_y"

        x_path = self._file_path(x_key)
        y_path = self._file_path(y_key)

        if not os.path.exists(x_path):
            raise FileNotFoundError(
                f"Data file not found: {x_path}. "
                "Re-initialize with download=True to fetch the dataset."
            )
        if not os.path.exists(y_path):
            raise FileNotFoundError(
                f"Label file not found: {y_path}. "
                "Re-initialize with download=True to fetch the dataset."
            )

        # Load raw numpy arrays
        X = np.load(x_path)  # shape: (n_trials, n_timesteps, n_channels)
        y = np.load(y_path)  # shape: (n_trials,)

        # Convert to torch tensors
        # X shape: (n_trials, n_timesteps, n_channels)
        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.long)

        assert len(self.data) == len(self.targets), (
            f"Mismatch between number of samples ({len(self.data)}) "
            f"and labels ({len(self.targets)})"
        )

    def __len__(self) -> int:
        """
        Return the number of EEG trials in the split.

        Returns:
            int: number of samples in the dataset

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single EEG trial and its label.

        Args:
            idx (int or list or torch.Tensor): index or indices of the
                sample(s) to retrieve.

        Returns:
            sample (torch.Tensor): EEG trial of shape
                ``(n_timesteps, n_channels)`` for a single index, or
                ``(batch, n_timesteps, n_channels)`` for a list/tensor of
                indices.
            target (torch.Tensor): class label(s), shape ``()`` for a
                single index or ``(batch,)`` for multiple indices.

        """
        if isinstance(idx, (list, torch.Tensor)):
            return self.data[idx], self.targets[idx]

        return self.data[idx], self.targets[idx]

    @property
    def n_timesteps(self) -> int:
        """Number of time steps per trial."""
        return self.data.shape[1]

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self.data.shape[2]

    @property
    def n_classes(self) -> int:
        """Number of unique class labels."""
        return int(self.targets.max().item()) + 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"n_trials={len(self)}, "
            f"n_channels={self.n_channels}, "
            f"n_timesteps={self.n_timesteps}, "
            f"n_classes={self.n_classes})"
        )
