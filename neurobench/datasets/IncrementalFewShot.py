from typing import Callable, List, Optional, Tuple, Union, Dict

from itertools import repeat, chain

import random
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, BatchSampler, RandomSampler

from tqdm import tqdm

from torch_mate.data.samplers import InfiniteClassSampler, DataSampler

import os
from torch import Tensor
from torchaudio.datasets.utils import _load_waveform


SAMPLE_RATE = 48000

class MSWC_query(Dataset):

    def __init__(self, walker):

        self._walker = walker

    def __getitem__(self, index: int):
        """ Getter method to get waveform samples.

        Args:
            idx (int): Index of the sample.

        Returns:
            sample (tensor): Individual waveform sample, padded to always match dimension (1, 48000).
            target (int): Corresponding keyword index based on FSCIL_KEYWORDS order (by decreasing number of samples in original dataset).
        """
        item = self._walker[index]

        dirname = item[0]
        waveform = _load_waveform(dirname, item[1], SAMPLE_RATE)

        if waveform.size()[1] != SAMPLE_RATE:
            full_size = torch.zeros((1,SAMPLE_RATE))
            full_size[:, :waveform.size()[1]] = waveform
            waveform = full_size


        return (waveform, item[2])

    def __len__(self):
        """ Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return(len(self._walker))


def get_indices_per_class(dataset: Dataset, support_query_split: Optional[Tuple[int, int]] = None, samples_per_class: Optional[int] = None) -> Union[Dict[int, List[int]], Dict[int, Tuple[List[int], List[int]]]]:
    indices_per_class = {}

    if not samples_per_class:
        for i, (_, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Getting indices per class"):
            if not isinstance(label, int):
                label = label.item()

            if label not in indices_per_class:
                indices_per_class[label] = []

            indices_per_class[label].append(i)
    else:
        for i in range(len(dataset) // samples_per_class):
            label = dataset[i*samples_per_class][1]
            indices_per_class[label] = list(range(i * samples_per_class, (i + 1) * samples_per_class))

    if support_query_split is not None:
        n_support, n_query = support_query_split

        for key in indices_per_class.keys():
            indices_per_class[key] = (indices_per_class[key][:n_support], indices_per_class[key][n_support:n_support+n_query])

    return indices_per_class


class IncrementalFewShot(IterableDataset):

    def __init__(self,
                 dataset: Dataset,
                 n_way: int,
                 k_shot: int,
                 query_shots: int = -1,
                 first_iter_ways_shots: Optional[Tuple[int, int]] = None,
                 support_query_split: Optional[Tuple[int, int]] = None,
                 incremental: bool = False,
                 cumulative: bool = False,
                 always_include_classes: Optional[List[int]] = None,
                 samples_per_class: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 per_class_transform: Optional[Callable] = None):
        """Dataset for few shot learning.

        Example usage:

        >>> # Few-shot learning
        >>> dataset = FewShot(dataset, n_way=5, k_shot=1)
        >>> # FSCIL (https://openaccess.thecvf.com/content/CVPR2022/papers/Chi_MetaFSCIL_A_Meta-Learning_Approach_for_Few-Shot_Class_Incremental_Learning_CVPR_2022_paper.pdf)
        >>> dataset = FewShot(dataset, n_way=5, k_shot=5, query_shots=50,
        >>>                   incremental=True, cumulative=True, 
        >>>                   first_iter_ways_shots=(20, 50),
        >>>                   support_query_split=(250, 250)
        >>> )
        >>> # N + M way, K shot learning (https://arxiv.org/abs/1812.10233)
        >>> dataset = FewShot(dataset, n_way=12, k_shot=1, always_include_classes=[0, 1]) # 10 + 2 way 1 shot learning

        Args:
            dataset (Dataset): The dataset to use. Labels should be integers or torch Scalars.
            n_way (int): Number of classes in the query and query set.
            k_shot (int, optional): Number of samples per class in the support set.
            query_shots (Optional[int]): Number of samples per class in the query set. If not set, query_shots is set to k_shot. Defaults to -1.
            first_iter_ways_shots (Optional[Tuple[int, int]], optional): Number of classes and samples per class for the first iteration. Defaults to None.
            support_query_split (Optional[Tuple[int, int]], optional): Create non-overlapping support and query pools of given number of samples per class. Defaults to None.
            incremental (bool, optional): Whether to incrementally sample classes. Defaults to False.
            cumulative (bool, optional): Whether to increase the query set size with each iteration. This flag will only work when incremental is set to True. Defaults to False.
            always_include_classes (Optional[List[int]], optional): List of classes to always include in both in the support and query set. Defaults to None.
            samples_per_class (Optional[int], optional): Number of samples per class to use. Can be used for large datasets where the classes are ordered (class_0_sample_0, c0s1, c0s2, c1s0, c1s1, c1s2, ...) to avoid iterating over the whole dataset for index per class computation. Defaults to None.
            transform (Optional[Callable], optional): Transform applied to every data sample. Will be reapplied every time a batch is served. Defaults to None.
            per_class_transform (Optional[Callable], optional): Transform applied to every data sample. Will be applied to every class separately. Defaults to None.
        """

        if cumulative and not incremental:
            raise ValueError(
                "cumulative can only be set to True when incremental is True."
            )

        if first_iter_ways_shots and not incremental:
            raise ValueError(
                "first_iter_ways_shots can only be set when incremental is True."
            )

        if always_include_classes is not None:
            if len(always_include_classes) > n_way:
                raise ValueError("always_include_classes cannot have more elements than n_way.")

            if len(set(always_include_classes)) != len(always_include_classes):
                raise ValueError("always_include_classes cannot have duplicate elements.")

            if cumulative:
                raise ValueError("always_include_classes cannot be used when cumulative is True.")

        self.dataset = dataset
        self.support_query_split = support_query_split
        self.indices_per_class = get_indices_per_class(self.dataset, self.support_query_split, samples_per_class)

        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots if query_shots != -1 else k_shot

        self.always_include_classes = always_include_classes
        self.first_iter_ways_shots = first_iter_ways_shots

        self.incremental = incremental
        self.cumulative = cumulative

        self.total_classes = len(self.indices_per_class)

        if self.incremental:

            self.classes_to_sample_from = list(set(self.indices_per_class.keys()))
        else:
            self.class_sampler = InfiniteClassSampler(self.total_classes, self.n_way)

    def reset(self):
        """Reset sampler for new iteration of dataset
        """

        if self.incremental:

            self.classes_to_sample_from = list(set(self.indices_per_class.keys()))
        else:
            self.class_sampler = InfiniteClassSampler(self.total_classes, self.n_way)


    def __iter__(self):
        """Get a batch of samples for a k-shot n-way task.

        Returns:
            tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: The support and query sets in the form of ((X_support, X_query), (y_support, y_query))
        """

        cumulative_classes = []

        # Change the way and shots for the first iteration
        if self.first_iter_ways_shots:
            first_iter_ways, k_shot = self.first_iter_ways_shots
            first_iter = random.sample(self.classes_to_sample_from, first_iter_ways)  
            self.classes_to_sample_from = list(filter(lambda index:index not in first_iter, self.classes_to_sample_from))

            if self.cumulative:
                cumulative_classes.extend(first_iter)

            # Yields first iteration
            out = self._inner_iter(first_iter, first_iter_ways, k_shot)
            yield out


        k_shot = self.k_shot

        self.class_sampler = BatchSampler(DataSampler(self.classes_to_sample_from, RandomSampler(self.classes_to_sample_from, replacement=False)), batch_size=self.n_way, drop_last=True)


        
        for new_class_indices in self.class_sampler:
            n_way = len(new_class_indices)

            class_indices = new_class_indices + cumulative_classes

            if self.cumulative:
                cumulative_classes.extend(new_class_indices)

            # Yields iterative sessions
            out = self._inner_iter(class_indices, n_way, k_shot)
            yield out


    def _inner_iter(self, iter_classes, n_way, k_shot):
        X_train_samples = []
        y_train_samples = []
        query_info = []

        if self.always_include_classes is not None:
            # This line also makes sure that the always include classes are always
            # in the same position in the batch
            iter_classes = list(
                set(iter_classes) - set(self.always_include_classes)
            )[:len(iter_classes) - len(self.always_include_classes
                                        )] + self.always_include_classes

        for i, class_index in enumerate(iter_classes):
            if self.support_query_split:
                within_class_indices = np.concatenate([np.random.choice(self.indices_per_class[class_index][j], shot, replace=False) for j, shot in [(0, k_shot), (1, self.query_shots)]])
            else:
                within_class_indices = np.random.choice(
                    self.indices_per_class[class_index],
                    k_shot + self.query_shots,
                    replace=False)

            local_class_index = i if not self.incremental else class_index

            # Only in the case of cumulative we need to make sure that we only
            # include the new classes in the support set and not the previous classes
            if i < n_way:
                support_samples = torch.stack(
                    [self.dataset[j][0] for j in within_class_indices[:k_shot]])
                y_train_samples.extend([local_class_index] * k_shot)
                X_train_samples.extend(support_samples)


            query_info += [(self.dataset[j][2], self.dataset[j][3], local_class_index) for j in within_class_indices[k_shot:]]


        query_set = MSWC_query(query_info)

        out = ((torch.stack(X_train_samples),torch.tensor(y_train_samples)),
                query_set, iter_classes)

        return out