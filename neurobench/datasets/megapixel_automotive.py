"""
"""

from .dataset import NeuroBenchDataset

class MegapixelAutomotive(NeuroBenchDataset):
    """
    """
    def __init__(self, path, split="testing"):
        ...
    
    def __len__(self):
        ...
    
    def __getitem__(self, idx):
        ...