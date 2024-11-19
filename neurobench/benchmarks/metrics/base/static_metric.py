from abc import ABC, abstractmethod
from neurobench.models import NeuroBenchModel


class StaticMetric(ABC):
    @abstractmethod
    def __call__(self, model: NeuroBenchModel) -> float:
        pass
