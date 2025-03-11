from abc import ABC
from neurobench.processors.abstract import (
    NeuroBenchPreProcessor,
    NeuroBenchPostProcessor,
)
from typing import List, Callable, Tuple
from torch import Tensor


class ProcessorManager(ABC):
    """
    Manager class for handling preprocessors and postprocessors.

    This class manages the list of preprocessing and postprocessing steps, ensuring
    that the appropriate processors are applied to data before and after the model
    processing. It supports replacing the lists of preprocessors and postprocessors
    and ensures that each processor is of the correct type.

    Attributes:
        preprocessors (List[NeuroBenchPreProcessor]): List of preprocessors to
        be applied before model inference.
        postprocessors (List[NeuroBenchPostProcessor]): List of postprocessors
        to be applied after model inference.

    """

    def __init__(
        self,
        preprocessors: List[
            NeuroBenchPreProcessor
            | Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
        ],
        postprocessors: List[NeuroBenchPostProcessor | Callable[[Tensor], Tensor]],
    ):
        """
        Initialize the ProcessorManager with the given preprocessors and postprocessors.

        Args:
            preprocessors (List[NeuroBenchPreProcessor]): List of preprocessors
            to be applied to the data.
            postprocessors (List[NeuroBenchPostProcessor]): List of postprocessors
            to be applied to the data.

        Raises:
            TypeError: If any element in preprocessors is not an instance of
            NeuroBenchPreProcessor, or if any element in postprocessors is not
            an instance of NeuroBenchPostProcessor.

        """

        if any(
            not (isinstance(p, NeuroBenchPreProcessor) or callable(p))
            for p in preprocessors
        ):
            raise TypeError(
                "All preprocessors must be instances of NeuroBenchPreProcessor or callable functions"
            )
        if any(
            not (isinstance(p, NeuroBenchPostProcessor) or callable(p))
            for p in postprocessors
        ):
            raise TypeError(
                "All postprocessors must be instances of NeuroBenchPostProcessor or callable functions"
            )

        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

    def replace_preprocessors(
        self,
        preprocessors: List[
            NeuroBenchPreProcessor
            | Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
        ],
    ):
        """
        Replace the current list of preprocessors with the provided list.

        Args:
            preprocessors (List[NeuroBenchPreProcessor]): List of new preprocessors
            to replace the current ones.

        Raises:
            TypeError: If any element in the provided list is not an instance of
            NeuroBenchPreProcessor.

        """
        if any(
            not (isinstance(p, NeuroBenchPreProcessor) or callable(p))
            for p in preprocessors
        ):
            raise TypeError(
                "All preprocessors must be instances of NeuroBenchPreProcessor or callable functions"
            )
        self.preprocessors = preprocessors

    def replace_postprocessors(
        self, postprocessors: List[NeuroBenchPostProcessor | Callable[[Tensor], Tensor]]
    ):
        """
        Replace the current list of postprocessors with the provided list.

        Args:
            postprocessors (List[NeuroBenchPostProcessor]): List of new postprocessors
            to replace the current ones.

        Raises:
            TypeError: If any element in the provided list is not an instance of
            NeuroBenchPostProcessor.

        """
        if any(
            not (isinstance(p, NeuroBenchPostProcessor) or callable(p))
            for p in postprocessors
        ):
            raise TypeError(
                "All postprocessors must be instances of NeuroBenchPostProcessor or callable functions"
            )
        self.postprocessors = postprocessors

    def preprocess(self, data: Tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Apply preprocessing steps to the input data.

        Applies each preprocessor in the list sequentially to the provided data.

        Args:
            data (tuple[Tensor, Tensor]): The data to preprocess.

        Returns:
            tuple[Tensor, Tensor]: The preprocessed data.

        """
        for preprocessor in self.preprocessors:
            data = preprocessor(data)
        return data

    def postprocess(self, data: Tensor) -> Tensor:
        """
        Apply postprocessing steps to the input data.

        Applies each postprocessor in the list sequentially to the provided data.

        Args:
            data (Tensor): The data to postprocess.

        Returns:
            Tensor: The postprocessed data.

        """
        for postprocessor in self.postprocessors:
            data = postprocessor(data)
        return data
