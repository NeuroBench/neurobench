from typing import List

import torch

from neurobench.data.utils import NeuroBenchClassificationDataset

def get_indices_for_phase(dataset: NeuroBenchClassificationDataset, phase: str, init_subjects: List[str]):
    indices = []

    for i in range(len(dataset)):
        subject = dataset.get_subject(i)

        if (subject in init_subjects and phase == "init") or (subject not in init_subjects and phase == "cont"):
            indices.append(i)

    return indices

class ContinualDomainAdaptationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: NeuroBenchClassificationDataset, init_subjects: List[str], phase: str):
        indices = self.get_indices_for_phase(dataset, phase, init_subjects)

        if phase == "cont":
            raise NotImplementedError("cont phase support with batches per subject is not implemented yet.")

        self.dataset = torch.utils.data.Subset(dataset, indices)

    def __getitem__(self, n: int):
        return self.dataset[n]
    
    def __len__(self):
        return len(self.dataset)
