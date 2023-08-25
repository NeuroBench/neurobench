
from neurobench.models.M5 import M5

from .model import *
from ..utils import _lazy_import

def SNNTorchModel(*args, **kwargs):
    return _lazy_import("neurobench.models", ".snntorch_models", "SNNTorchModel")(*args, **kwargs)

def TorchModel(*args, **kwargs):
    return _lazy_import("neurobench.models", ".torch_model", "TorchModel")(*args, **kwargs)
