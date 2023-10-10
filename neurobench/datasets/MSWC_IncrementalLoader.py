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
from neurobench.datasets.MSWC import MSWC
from neurobench.datasets.MSWC import MSWC_query


SAMPLE_RATE = 48000


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
                 support_query_split: Optional[Tuple[int, int]] = None,
                 samples_per_class: Optional[int] = None):
        """Dataset for few shot learning.

        Args:
            n_way (int): Number of classes in the query and query set.
            k_shot (int, optional): Number of samples per class in the support set.
            root (str): Path of the folder where to find the dataset language folders.
            inc_languages (List[str], optional): List of languages 2 letters names to use as incremental sessions. 
            query_shots (Optional[int]): Number of samples per class in the query set. If not set, query_shots is set to k_shot. Defaults to -1.
            support_query_split (Optional[Tuple[int, int]], optional): Create non-overlapping support and query pools of given number of samples per class. Defaults to None.
            samples_per_class (Optional[int], optional): Number of samples per class to use. Can be used for large datasets where the classes are ordered (class_0_sample_0, c0s1, c0s2, c1s0, c1s1, c1s2, ...) to avoid iterating over the whole dataset for index per class computation. Defaults to None.
        """

        self.support_query_split = support_query_split
        self.indices_per_lang = get_indices_per_class(inc_languages, root, self.support_query_split, samples_per_class)

        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots if query_shots != -1 else k_shot

        self.languages = inc_languages
        self.root = root

    # def reset(self):
    #     """Reset sampler for new iteration of dataset
    #     """
    #     self.classes_to_sample_from = list(set(self.indices_per_class.keys()))


    def __iter__(self):
        """Get a batch of samples for a k-shot n-way task.

        Returns:
            tuple[List[tuple[Tensor, Tensor]], Dataset, List[int]]: The support as a list of all shots, the query as a Dataset object and the list of cumulative query classes.
        """

        cumulative_classes = {}

        k_shot = self.k_shot

        self.cumulative_query = []
        for lang in random.sample(self.languages, len(self.languages)):
            dataset = MSWC(root=self.root, subset="evaluation", language=lang)

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