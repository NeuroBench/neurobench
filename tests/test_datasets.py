import os

from neurobench.datasets import SpeechCommands
from neurobench.datasets import MegapixelAutomotive
from neurobench.datasets import PrimateReaching
from neurobench.datasets import MackeyGlass

import torch

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

def test_mackey_glass():
    mg = MackeyGlass(17, 0.9)

    assert len(mg) > 0

    trainset = torch.utils.data.Subset(mg, mg.ind_train)
    testset = torch.utils.data.Subset(mg, mg.ind_test)

    assert len(trainset) == mg.traintime_pts
    assert len(testset) == mg.testtime_pts

    assert trainset[0][0].shape == (1,1)
    assert trainset[0][1].shape == (1,)
    assert testset[0][0].shape == (1,1)
    assert testset[0][1].shape == (1,)

    assert(torch.eq(trainset[0][0], mg[0][0]))
    assert(torch.eq(trainset[0][1], mg[0][1]))
    assert(torch.eq(testset[0][0], mg[mg.traintime_pts][0]))
    assert(torch.eq(testset[0][1], mg[mg.traintime_pts][1]))