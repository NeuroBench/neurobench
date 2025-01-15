from abc import ABC, abstractmethod
from torch import Tensor
from neurobench.models import NeuroBenchModel


class WorkloadMetric(ABC):

    def __init__(self, requires_hooks: bool = False):
        self._requires_hooks = requires_hooks

    @abstractmethod
    def __call__(
        self, model: NeuroBenchModel, preds: Tensor, data: tuple[Tensor, Tensor]
    ) -> float:
        pass

    @property
    def requires_hooks(self) -> bool:
        return self._requires_hooks


class AccumulatedMetric(WorkloadMetric):

    def __init__(self, requires_hooks: bool = False):
        super().__init__(requires_hooks)

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
