import os

from neurobench.datasets import SpeechCommands
from neurobench.datasets import MegapixelAutomotive
from neurobench.datasets import PrimateReaching

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