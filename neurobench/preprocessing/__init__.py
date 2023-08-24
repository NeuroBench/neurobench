from .preprocessor import *
from ..utils import _lazy_import

def S2SProcessor(*args, **kwargs):
    return _lazy_import("neurobench.preprocessing", ".speech2spikes", "S2SProcessor")(*args, **kwargs)

def MFCCProcessor(*args, **kwargs):
    return _lazy_import("neurobench.preprocessing", ".mfcc", "MFCCProcessor")(*args, **kwargs)
