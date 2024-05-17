from .dataset import *
from .utils import *
from ..utils import _lazy_import


def SpeechCommands(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".speech_commands", "SpeechCommands")(
        *args, **kwargs
    )


def PrimateReaching(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".primate_reaching", "PrimateReaching")(
        *args, **kwargs
    )


def Gen4DetectionDataLoader(*args, **kwargs):
    return _lazy_import(
        "neurobench.datasets", ".megapixel_automotive", "Gen4DetectionDataLoader"
    )(*args, **kwargs)


def MackeyGlass(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".mackey_glass", "MackeyGlass")(
        *args, **kwargs
    )


def WISDM(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".WISDM", "WISDM")(*args, **kwargs)


def MSWC(*args, **kwargs):
    return _lazy_import("neurobench.datasets", ".MSWC_dataset", "MSWC")(*args, **kwargs)
