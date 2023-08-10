import unittest
from neurobench.datasets.mackey_glass_cleaned import MackeyGlass_clean

import torch

def test_mackey_glass():
    mg = MackeyGlass_clean(17, 0.9)

    from torch.utils.data import Subset
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