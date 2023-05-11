from typing import Union, Optional, Callable
from pathlib import Path

import torch

class NeuroBenchDataset(torch.utils.data.Dataset):
    """Base class for all NeuroBench datasets.
    """
    def __init__(self,
                root: Union[str, Path],
                download: bool = False,
                subset: Optional[str] = None,
                transform: Union[Callable, None] = None) -> None:
        raise NotImplementedError("NeuroBenchDataset is an abstract class and should not be instantiated directly.")
