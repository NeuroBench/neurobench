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
from neurobench.datasets.MSWC_multilingual import MSWC


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


def get_indices_per_class(languages, root, support_query_split: Optional[Tuple[int, int]] = None, samples_per_class: Optional[int] = None) -> Union[Dict[int, List[int]], Dict[int, Tuple[List[int], List[int]]]]:
    
    indices_per_lang = {}
    for lang in languages:
        indices_per_lang[lang] = {}
        indices_per_class = indices_per_lang[lang]
        dataset = MSWC(root=root, subset="evaluation", language=lang)

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
        del dataset

    return indices_per_lang


class IncrementalFewShot(IterableDataset):

    def __init__(self,
                 n_way: int,
                 k_shot: int,
                 root: str,
                 inc_languages: list = ['fa', 'eo', 'pt', 'eu', 'pl', 'cy', 'nl', 'ru', 'es', 'it'],
                 query_shots: int = -1,
                 first_iter_ways_shots: Optional[Tuple[int, int]] = None,
                 support_query_split: Optional[Tuple[int, int]] = None,
                 incremental: bool = True,
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

        self.support_query_split = support_query_split
        self.indices_per_lang = get_indices_per_class(inc_languages, root, self.support_query_split, samples_per_class)

        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots if query_shots != -1 else k_shot

        self.always_include_classes = always_include_classes
        self.first_iter_ways_shots = first_iter_ways_shots

        self.incremental = incremental
        self.cumulative = cumulative

        self.languages = inc_languages
        self.root = root

    # def reset(self):
    #     """Reset sampler for new iteration of dataset
    #     """
    #     self.classes_to_sample_from = list(set(self.indices_per_class.keys()))


    def __iter__(self):
        """Get a batch of samples for a k-shot n-way task.

        Returns:
            tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: The support and query sets in the form of ((X_support, X_query), (y_support, y_query))
        """

        cumulative_classes = {}

        k_shot = self.k_shot

        self.cumulative_query = []
        # cumulative_datasets = {}
        for lang in random.sample(self.languages, len(self.languages)):
            dataset = MSWC(root=self.root, subset="evaluation", language=lang)
            # cumulative_datasets[lang] = dataset

            support_classes = random.sample(self.indices_per_lang[lang].keys(), self.n_way)
            cumulative_classes[lang] = support_classes

            # Yields iterative sessions
            out = self._inner_iter(lang, dataset, support_classes, cumulative_classes, self.n_way, k_shot)
            del dataset
            yield out


    def _inner_iter(self, language, dataset, support_classes, cumulative_classes, n_way, k_shot):
        X_train_samples = [[] for _ in range(k_shot)]
        y_train_samples = [[] for _ in range(k_shot)]

        for i, class_index in enumerate(support_classes):
            support_indices = np.random.choice(self.indices_per_lang[language][class_index][0], k_shot, replace=False)
            for i, index in enumerate(support_indices):
                data, real_class, _, _ = dataset[index]
                y_train_samples[i].append(real_class)
                X_train_samples[i].append(data)
            
            
        for i, class_index in enumerate(cumulative_classes[language]):
            query_indices = np.random.choice(self.indices_per_lang[language][class_index][1], self.query_shots, replace=False)

            self.cumulative_query += [(dataset[j][2], dataset[j][3], class_index) for j in query_indices]


        support = []
        for x, y in zip(X_train_samples, y_train_samples):
            shot = (torch.stack(x),torch.tensor(y, dtype=torch.long))
            support.append(shot)

        query_set = MSWC_query(self.cumulative_query)
        query_classes = []
        for lang_class in cumulative_classes.values():
            query_classes.extend(lang_class)


        out = (support, query_set, query_classes)

        return out