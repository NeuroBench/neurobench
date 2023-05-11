from typing import List, Tuple

import torch

from neurobench.data.utils.NeuroBenchDataset import NeuroBenchDataset
from torch_mate.data.utils import FewShot

def get_indices_for_init_phase(dataset: NeuroBenchDataset, init_labels: List[int]):
    indices = []

    for i, (_, label) in enumerate(dataset):
        if label in init_labels:
            indices.append(i)

    return indices


class IncrementalFewShotDataset:
    def __new__(self, dataset: NeuroBenchDataset, init_labels: List[int], way: int = 5, shot: int = 1) -> Tuple[NeuroBenchDataset, FewShot]:
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
            init_labels (List[int]): v of labels for initial phase.
            way (int, optional): Ways for few-shot continual phase. Defaults to 5.
            shot (int, optional): Shots for few-shot continual phase. Defaults to 1.

        Returns:
            Tuple[NeuroBenchDataset, FewShot]: Datasets for both init and cont phases.
        """

        init_indices = get_indices_for_init_phase(dataset, init_labels)
        init_subset = torch.utils.data.Subset(dataset, init_indices)

        cont_indices =  set(list(range(len(dataset)))) - set(init_subset)
        cont_subset = torch.utils.data.Subset(dataset, cont_indices)

        return init_subset, FewShot(cont_subset, way, shot, shot)
