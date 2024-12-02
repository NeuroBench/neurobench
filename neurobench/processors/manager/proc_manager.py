from abc import ABC
from neurobench.processors.abstract import (
    NeuroBenchPreProcessor,
    NeuroBenchPostProcessor,
)
from typing import List


class ProcessorManager(ABC):

    def __init__(
        self,
        preprocessors: List[NeuroBenchPreProcessor],
        postprocessors: List[NeuroBenchPostProcessor],
    ):

        if any(not isinstance(p, NeuroBenchPreProcessor) for p in preprocessors):
            raise TypeError(
                "All preprocessors must be instances of NeuroBenchPreProcessor"
            )
        if any(not isinstance(p, NeuroBenchPostProcessor) for p in postprocessors):
            raise TypeError(
                "All postprocessors must be instances of NeuroBenchPostProcessor"
            )

        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

    def replace_preprocessors(self, preprocessors: List[NeuroBenchPreProcessor]):
        if any(not isinstance(p, NeuroBenchPreProcessor) for p in preprocessors):
            raise TypeError(
                "All preprocessors must be instances of NeuroBenchPreProcessor"
            )
        self.preprocessors = preprocessors

    def replace_postprocessors(self, postprocessors: List[NeuroBenchPostProcessor]):
        if any(not isinstance(p, NeuroBenchPostProcessor) for p in postprocessors):
            raise TypeError(
                "All postprocessors must be instances of NeuroBenchPostProcessor"
            )
        self.postprocessors = postprocessors

    def preprocess(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor(data)
        return data

    def postprocess(self, data):
        for postprocessor in self.postprocessors:
            data = postprocessor(data)
        return data
