"""
This file contains code based on torch-mate: https://github.com/V0XNIHILI/torch-mate
"""

from typing import List, Optional, Tuple, Union, Dict

import random
import numpy as np
import torch
from torch.utils.data import IterableDataset

from neurobench.datasets.MSWC_dataset import MSWC, MSWC_query


def get_indices_per_class(
    languages,
    root,
    samples_per_class: int,
    support_query_split: Optional[Tuple[int, int]] = None,
) -> Union[Dict[int, List[int]], Dict[int, Tuple[List[int], List[int]]]]:
    indices_per_lang = {}
    for lang in languages:
        indices_per_lang[lang] = {}
        indices_per_class = indices_per_lang[lang]
        dataset = MSWC(root=root, subset="evaluation", language=lang)

        # Ordering samples index per class using the fact that all samples from one class are indexed together
        for i in range(len(dataset) // samples_per_class):
            label = dataset[i * samples_per_class][1]
            indices_per_class[label] = list(
                range(i * samples_per_class, (i + 1) * samples_per_class)
            )

        if support_query_split is not None:
            n_support, n_query = support_query_split

            for key in indices_per_class.keys():
                indices_per_class[key] = (
                    indices_per_class[key][:n_support],
                    indices_per_class[key][n_support : n_support + n_query],
                )
        del dataset

    return indices_per_lang


class IncrementalFewShot(IterableDataset):
    def __init__(
        self,
        k_shot: int,
        root: str,
        inc_languages: list = [
            "fa",
            "eo",
            "pt",
            "eu",
            "pl",
            "cy",
            "nl",
            "ru",
            "es",
            "it",
        ],
        query_shots: int = -1,
        support_query_split: Optional[Tuple[int, int]] = None,
    ):
        """
        Dataset for few shot learning.

        Args:
            k_shot (int, optional): Number of samples per class in the support set.
            root (str): Path of the folder where to find the dataset language folders.
            inc_languages (List[str], optional): List of languages 2 letters names to use as incremental sessions.
            query_shots (Optional[int]): Number of samples per class in the query set. If not set, query_shots is set to k_shot. Defaults to -1.
            support_query_split (Optional[Tuple[int, int]], optional): Create non-overlapping support and query pools of given number of samples per class. Defaults to None.

        """

        self.support_query_split = support_query_split
        self.samples_per_class = 200  # Number of samples per class to use. Used to simplify samples class indexing as the used dataset is ordered (class_0_sample_0, c0s1, c0s2, c1s0, c1s1, c1s2, ...).
        self.indices_per_lang = get_indices_per_class(
            inc_languages, root, self.samples_per_class, self.support_query_split
        )

        self.n_way = 10  # Number of classes in the support and support sets. Fixed based on the dataset
        self.k_shot = k_shot
        self.query_shots = query_shots if query_shots != -1 else k_shot

        self.languages = inc_languages
        self.root = root

    def __iter__(self):
        """
        Get a batch of samples for a k-shot n-way task.

        Returns:
            tuple[List[tuple[Tensor, Tensor]], Dataset, List[int]]: The support as a list of all shots, the query as a Dataset object and the list of cumulative query classes.

        """

        cumulative_classes = {}
        self.cumulative_query = []

        for lang in random.sample(self.languages, len(self.languages)):
            dataset = MSWC(root=self.root, subset="evaluation", language=lang)

            support_classes = random.sample(
                sorted(self.indices_per_lang[lang].keys()), self.n_way
            )
            cumulative_classes[lang] = support_classes

            # Yields iterative sessions
            out = self._inner_iter(lang, dataset, support_classes, cumulative_classes)
            del dataset
            yield out

    def _inner_iter(self, language, dataset, support_classes, cumulative_classes):
        for class_index in cumulative_classes[language]:
            query_indices = np.random.choice(
                self.indices_per_lang[language][class_index][1],
                self.query_shots,
                replace=False,
            )

            self.cumulative_query += [
                (dataset[j][3], class_index, dataset[j][2]) for j in query_indices
            ]

        query_set = MSWC_query(self.cumulative_query)

        X_train_samples = [[] for _ in range(self.k_shot)]
        y_train_samples = [[] for _ in range(self.k_shot)]

        # For every class
        for class_index in support_classes:
            support_indices = np.random.choice(
                self.indices_per_lang[language][class_index][0],
                self.k_shot,
                replace=False,
            )

            # For every (support) shot/sample
            for j, index in enumerate(support_indices):
                data, real_class, _, _ = dataset[index]
                y_train_samples[j].append(real_class)
                X_train_samples[j].append(data)

        support = []
        for x, y in zip(X_train_samples, y_train_samples):
            shot = (torch.stack(x), torch.tensor(y, dtype=torch.long))
            support.append(shot)

        query_classes = []
        for lang_class in cumulative_classes.values():
            query_classes.extend(lang_class)

        out = (support, query_set, query_classes)

        return out
