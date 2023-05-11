from typing import List, Union

import torch

from neurobench.data.utils import NeuroBenchDataset
from torch_mate.data.utils import FewShot

def get_indices_for_phase(dataset: NeuroBenchDataset, phase: str, init_labels: List[int]):
    indices = []

    for i, (_, label) in enumerate(dataset):
        if (label in init_labels and phase == "init") or (label not in init_labels and phase == "cont"):
            indices.append(i)

    return indices


class IncrementalFewShotDataset:
    def __new__(self, dataset: NeuroBenchDataset, init_labels: List[int], phase: str, way: int = 5, shot: int = 1) -> Union[NeuroBenchDataset, FewShot]:
        """This benchmark evaluates the ability of a model to perform
        class-incremental learning over its lifetime, where the model
        is required to learn new keywords or gestures as they are
        introduced. The dataset is initially divided into Train_init and
        Train_cont sets based on the classes of keywords or gestures,
        where Train_init represents the initial set of classes that the
        model is pre-trained on. The remaining classes are included in
        the Train_cont set. Each task in Train_cont is represented by
        a N*K sample batch (N-way K-shot) with unique keywords
        or gestures for every task. The model trains sequentially over
        these batches with each task introducing N new keywords or
        gestures to learn. The test set consists of unseen samples of
        all the keywords or gestures that have been exposed.

        Args:
            dataset (NeuroBenchDataset): Dataset to be used for incremental few-shot learning.
            init_labels (List): List of labels for initial phase.
            phase (str): Phase of the dataset. Can be either "init" (initial) or "cont" (continual).
            way (int, optional): Ways for few-shot continual phase. Defaults to 5.
            shot (int, optional): Shots for few-shot continual phase. Defaults to 1.

        Raises:
            ValueError: If phase is not "init" or "cont".

        Returns:
            Union[NeuroBenchDataset, FewShot]: Dataset for the given phase.
        """

        indices = get_indices_for_phase(dataset, phase, init_labels)
        subset = torch.utils.data.Subset(dataset, indices)

        if phase == "init":
            return subset
        elif phase == "cont":
            # TODO: add support for the fact that every batch, we want to get N new keywords
            return FewShot(subset, way, shot, shot)
        else:
            raise ValueError(f"Unknown phase: {phase}. Valid phases are: 'init' and 'cont'.")
