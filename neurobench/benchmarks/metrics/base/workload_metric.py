from abc import ABC, abstractmethod
from torch import Tensor


class WorkloadMetric(ABC):
    @abstractmethod
    def __call__(self, model, preds: Tensor, data: Tensor):
        pass
