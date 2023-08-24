from . import NeuroBenchProcessor
import torch
from tqdm import tqdm
import numpy as np

class PrimateReachingProcessor(NeuroBenchProcessor):
    def __init__(self):
        raise NotImplementedError("The functionalities that was implemented here have been moved to neurobench.datasets.primate_reaching")

    def __call__(self, mode):
        raise NotImplementedError("The functionalities that was implemented here have been moved to neurobench.datasets.primate_reaching")
 