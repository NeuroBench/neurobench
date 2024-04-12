from .preprocessor import *
from ..utils import _lazy_import


def S2SPreProcessor(*args, **kwargs):
    return _lazy_import(
        "neurobench.preprocessing", ".speech2spikes", "S2SPreProcessor"
    )(*args, **kwargs)


def MFCCPreProcessor(*args, **kwargs):
    return _lazy_import("neurobench.preprocessing", ".mfcc", "MFCCPreProcessor")(
        *args, **kwargs
    )
