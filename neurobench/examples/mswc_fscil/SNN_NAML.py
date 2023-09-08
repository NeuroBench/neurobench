#
# NOTE: This task is still under development.
#

import torch

from speech2spikes import S2S

from neurobench.datasets import MSWC_FSCIL
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

## Define model ##
class SNN_MAML(TorchModel):
    def __init__(self, net):
        self.net = net

    def __call__(self, data):
        ...
