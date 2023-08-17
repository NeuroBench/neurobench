"""
"""

import torch

from glob import glob
from torchaudio import load

from .dataset import NeuroBenchDataset

class SpeechCommands(NeuroBenchDataset):
    """
    """
    def __init__(self, path, split="testing"):
        if split == "training":
            self.filenames = set(glob("*/*.wav", root_dir=path))
            with open(path+"validation_list.txt") as f:
                self.filenames -= set([n.strip() for n in f.readlines()])
            with open(path+"testing_list.txt") as f:
                self.filenames -= set([n.strip() for n in f.readlines()])
            self.filenames = sorted(list(self.filenames))
        elif split == "validation":
            with open(path+"validation_list.txt") as f:
                self.filenames = sorted([n.strip() for n in f.readlines()])
        else: 
            with open(path+"testing_list.txt") as f:
                self.filenames = sorted([n.strip() for n in f.readlines()])

        self.path = path
        self.labels = sorted(glob("*/", root_dir=path))
        self.labels = {key[:-1]: idx for idx, key in enumerate(self.labels)}

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        label = self.filenames[idx].split('/')[0]
        return load(self.path + self.filenames[idx])[0], self.label_to_index(label)

    def label_to_index(self, label):
        return torch.tensor(self.labels[label])