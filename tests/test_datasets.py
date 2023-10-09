import os

from neurobench.datasets import SpeechCommands
from neurobench.datasets import Gen4DetectionDataLoader
from neurobench.datasets import PrimateReaching
from neurobench.datasets import DVSGesture
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
    assert list(ds[0][0].shape) == [16000, 1] # timesteps, channels
    assert int(ds[0][1]) == 0

def test_dvs_gesture():
    path = dataset_path + "dvs_gesture/"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")
    ds = DVSGesture(path)

    assert len(ds) > 0
    assert list(ds[0][0].shape) == [340, 3, 128, 128]
   
    assert int(ds[0][1]) >= 0
    assert int(ds[0][1]) <= 10

def test_mackey_glass():
    mg = MackeyGlass(17,197,0.7206597)

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

    bin_window = 3
    mg = MackeyGlass(17,197,0.7206597, bin_window=bin_window)
    trainset = torch.utils.data.Subset(mg, mg.ind_train)
    testset = torch.utils.data.Subset(mg, mg.ind_test)
    
    assert trainset[0][0].shape == (bin_window,1)
    assert trainset[0][1].shape == (1,)
    assert testset[0][0].shape == (bin_window,1)
    assert testset[0][1].shape == (1,)
    
    assert(torch.eq(mg[0][1], mg[1][0][-1])) # ensure target from previous timestep is appended to lookback window


def test_1mp():
    path = dataset_path + "Gen 4 Histograms/"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")

    dl = Gen4DetectionDataLoader(
        dataset_path=path,
        split="testing",
        label_map_path="neurobench/datasets/label_map_dictionary.json",
        batch_size = 1,
        num_tbins = 12,
        preprocess_function_name="histo",
        delta_t=50000,
        channels=2,  # histograms have two channels
        height=360,
        width=640,
        max_incr_per_pixel=5,
        class_selection=["pedestrian", "two wheeler", "car"],
        num_workers=1)

    assert len(dl) > 0
    assert iter(dl) is not None
    assert next(iter(dl)) is not None

    data = next(iter(dl))
    assert len(data) == 3
    assert data[0].shape == (1, 12, 2, 360, 640) # batch, timestep, channel, height, width
    assert isinstance(data[1], list) # list[list[nparr]]
    assert isinstance(data[2], dict)

def test_primate_reaching():
    path = dataset_path + "primate_reaching/PrimateReachingDataset"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")

    dataset = PrimateReaching(file_path=path,
                              filename="indy_20170131_02.mat",
                              num_steps=250, train_ratio=0.8, bin_width=0.004,
                              biological_delay=50)

    # check dataset non-empty
    assert len(dataset)

    train_set_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, dataset.ind_train), batch_size=1, shuffle=False)
    test_set_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, dataset.ind_test), batch_size=1, shuffle=False)

    # correct amount of samples
    assert len(train_set_loader) == len(dataset.ind_train)
    assert len(test_set_loader) == len(dataset.ind_test)

    # correct shapes
    assert next(iter(train_set_loader))[0].shape == (1, 250, 96)
    assert next(iter(train_set_loader))[1].shape == (1, 2)
    assert next(iter(test_set_loader))[0].shape == (1, 250, 96)
    assert next(iter(test_set_loader))[1].shape == (1, 2)