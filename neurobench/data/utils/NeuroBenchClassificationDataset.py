from typing import Tuple

from torch import Tensor

from .NeuroBenchDataset import NeuroBenchDataset

class NeuroBenchClassificationDataset(NeuroBenchDataset):
    """Base class for the datasets to be used for incremental few-shot classification and
    continual domain adaptation.
    """
    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        raise NotImplementedError
    
    def get_subject(self, n: int) -> str:
        raise NotImplementedError
    
    # TODO: add support for the fact that the test set is changing over time (see section 4.1.1 of the paper)
