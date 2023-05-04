from typing import Union, Optional, Callable
from pathlib import Path

import torch

from .SPEECHCOMMANDS_35C import SPEECHCOMMANDS_35C

INIT_KEYWORDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class IncrementalFewShotSC:
    def __new__(self, phase: str, root: Union[str, Path], download: bool = False, subset: Optional[str] = None, transform: Union[Callable, None] = None):
        if phase == "init":
            dataset = SPEECHCOMMANDS_35C(root=root, download=download, subset=subset, transform=transform)

            indices = []

            for _, label in dataset:
                if label in INIT_KEYWORDS:
                    indices.append(label)

            return torch.utils.data.Subset(dataset, indices)
        elif phase == "cont":
            raise NotImplementedError("IncrementalFewShotSC is not yet implemented for phase 'cont'.")
        else:
            raise ValueError(f"Unknown phase: {phase}. Valid phases are: 'init' and 'cont'.")

