from neurobench.datasets.dataset import Dataset
import numpy as np
import torch
import os
from .utils import download_url
from urllib.error import URLError

"""

The original WISDM dataset contains motion sensor recordings from participants
performing daily activities (walking, jogging, sitting, standing, etc.) using a
smartwatch. This version provides preprocessed train/validation splits ready for
use in the NeuroBench THOR benchmark.

Data is automatically downloaded from:
https://huggingface.co/datasets/neuromorphic-polito/siddha

Original dataset reference:
    Weiss, G. M., et al. (2019). Smartphone and Smartwatch-Based Biometrics Using
    Activities of Daily Living. IEEE Access, 7, 133190–133202.
    https://doi.org/10.1109/ACCESS.2019.2940729
"""

BASE_URL = "https://huggingface.co/datasets/neuromorphic-polito/siddha/resolve/main"

FILES = {
    "data": "subset_watch_2s_gho.npz",
}

_NPZ_SPLIT_KEYS = {
    "train": ("arr_0", "arr_3"),
    "val": ("arr_1", "arr_4"),
    "test": ("arr_2", "arr_5"),
}


class WISDM(Dataset):
    """
    Preprocessed WISDM smartwatch activity recognition dataset adapted for the THOR
    challenge.

    The dataset contains windowed tri-axial accelerometer and gyroscope recordings
    from a smartwatch, segmented into fixed-length trials with shape
    ``(n_trials, n_timesteps, n_channels)``. Each sample is a single activity
    window and the target is an integer class label encoding one of the six
    recognised daily activities.

    Args:
        root (str): Root directory where the dataset file is stored (or
            will be downloaded to).
        split (str): Which split to load. One of ``"train"``, ``"val"``,
            or ``"test"``.
        download (bool): If ``True``, downloads the dataset file from
            HuggingFace if it is not already present in ``root``.

    Reference:
        Weiss, G. M., et al. (2019). Smartphone and Smartwatch-Based Biometrics
        Using Activities of Daily Living. IEEE Access, 7, 133190–133202.
        https://doi.org/10.1109/ACCESS.2019.2940729

    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
    ):
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.root = root
        self.split = split

        os.makedirs(self.root, exist_ok=True)

        if download:
            self._download()

        self._load_data()

    def _file_path(self) -> str:
        """Return the full local path for the dataset file."""
        return os.path.join(self.root, FILES["data"])

    def _download(self):
        """Download the dataset file if not already present."""
        dest = self._file_path()
        if os.path.exists(dest):
            return
        url = f"{BASE_URL}/{FILES['data']}"
        try:
            print(f"Downloading {url}")
            download_url(url, dest)
        except URLError as error:
            raise RuntimeError(
                f"Failed to download {FILES['data']}:\n{error}"
            ) from error

    def _load_data(self):
        """Load the appropriate split into tensors."""
        path = self._file_path()

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}. "
                "Re-initialize with download=True to fetch the dataset."
            )

        npz = np.load(path)

        x_key, y_key = _NPZ_SPLIT_KEYS[self.split]

        X = npz[x_key]  # shape: (n_trials, n_timesteps, n_channels)
        y = npz[y_key]  # shape: (n_trials,) or (n_trials, n_classes) one-hot

        # Collapse one-hot labels if necessary
        if y.ndim == 2:
            y = np.argmax(y, axis=-1)

        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.long)

        assert len(self.data) == len(self.targets), (
            f"Mismatch between number of samples ({len(self.data)}) "
            f"and labels ({len(self.targets)})"
        )

    def __len__(self) -> int:
        """
        Return the number of activity windows in the split.

        Returns:
            int: number of samples in the dataset

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single activity window and its label.

        Args:
            idx (int or list or torch.Tensor): index or indices of the
                sample(s) to retrieve.

        Returns:
            sample (torch.Tensor): sensor window of shape
                ``(n_timesteps, n_channels)`` for a single index, or
                ``(batch, n_timesteps, n_channels)`` for a list/tensor of
                indices.
            target (torch.Tensor): class label(s), shape ``()`` for a
                single index or ``(batch,)`` for multiple indices.

        """
        return self.data[idx], self.targets[idx]

    @property
    def n_timesteps(self) -> int:
        """Number of time steps per trial."""
        return self.data.shape[1]

    @property
    def n_channels(self) -> int:
        """Number of sensor channels."""
        return self.data.shape[2]

    @property
    def n_classes(self) -> int:
        """Number of unique activity classes."""
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
