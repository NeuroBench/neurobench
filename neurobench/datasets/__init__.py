from .dataset import *
from ..utils import _lazy_import

def SpeechCommands(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".speech_commands", "SpeechCommands")(*args, **kwargs)

def PrimateReaching(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".primate_reaching", "PrimateReaching")(*args, **kwargs)

def Gen4DetectionDataLoader(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".megapixel_automotive", "Gen4DetectionDataLoader")(*args, **kwargs)

def DVSGesture(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".DVSGesture_loader", "DVSGesture")(*args, **kwargs)

def MackeyGlass(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".mackey_glass", "MackeyGlass")(*args, **kwargs)

def WISDMDataLoader(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".WISDM_loader", "WISDMDataLoader")(*args, **kwargs)

