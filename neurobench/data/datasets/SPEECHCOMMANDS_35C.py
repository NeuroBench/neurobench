from torch import Tensor
import torch
from torchaudio.datasets import SPEECHCOMMANDS

from typing import Union, Optional, Callable, Tuple
from pathlib import Path



class SPEECHCOMMANDS_35C(SPEECHCOMMANDS):
    def __init__(self,
                root: Union[str, Path],
                download: bool = False,
                subset: Optional[str] = None,
                transform: Union[Callable, None] = None) -> None:
        super().__init__(root=root, url="speech_commands_v0.02", download=download, subset=subset)

        self.transform = transform

        all_meta_data = [self.get_metadata(n) for n in range(len(self))]
        self.labels = sorted(list(set([m[2] for m in all_meta_data])))

    def label_to_index(self, word: str) -> int:
        return torch.tensor(self.labels.index(word))

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        waveform, _, label, _, _ = super().__getitem__(n)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, self.label_to_index(label)
