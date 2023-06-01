from torch import Tensor
import torch
from torchaudio.datasets import SPEECHCOMMANDS

from typing import Union, Optional, Callable
from pathlib import Path

from neurobench.data.utils import NeuroBenchClassificationDataset


class SPEECHCOMMANDS_35C(NeuroBenchClassificationDataset):
    def __init__(self,
                root: Union[str, Path],
                download: bool = False,
                subset: Optional[str] = None,
                transform: Union[Callable, None] = None) -> None:
        """Google Speech Commands V2 dataset.

        Args:
            root (Union[str, Path]): Root directory where datasets exist or will be saved.
            download (bool, optional): Whether to download the dataset to disk. Defaults to False.
            subset (Optional[str], optional): Which subset of the dataset to use. Can be either "training"/None, "validation" or "testing". Defaults to None.
            transform (Union[Callable, None], optional): Transform to apply to data. Defaults to None.
        """

        self.dataset = SPEECHCOMMANDS(root=root, url="speech_commands_v0.02", download=download, subset=subset)

        self.transform = transform

        all_metadata = [self.dataset.get_metadata(n) for n in range(len(self))]
        self.labels = sorted(list(set([m[2] for m in all_metadata])))

    def label_to_index(self, word: str) -> int:
        return torch.tensor(self.labels.index(word))

    def __getitem__(self, n: int):
        waveform, _, label, _, _ = self.dataset[n]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, self.label_to_index(label)
    
    def get_subject(self, n: int) -> str:
        return self.dataset.get_metadata(n)[3]
    
    def __len__(self):
        return len(self.dataset)
