import os

from neurobench.datasets import SpeechCommands
from neurobench.datasets import MegapixelAutomotive
from neurobench.datasets import PrimateReaching
from neurobench.datasets.DVSGesture_loader import DVSGesture

dataset_path = "data/"

def test_speech_commands():
    path = dataset_path + "speech_commands/"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")
    ds = SpeechCommands(path)

    assert len(ds) > 0
    assert ds[0][0].shape
    assert list(ds[0][0].shape) == [1, 16000]
    assert int(ds[0][1]) == 0

def test_dvs_gesture():
    dataset_path = 'neurobench/datasets/'
    path = dataset_path + "DVSGesture/"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")
    ds = DVSGesture(path)

    assert len(ds) > 0
    assert list(ds[0][0].shape) == [340, 3, 128, 128]
   
    assert int(ds[0][1]) >= 0
    assert int(ds[0][1]) <= 10

