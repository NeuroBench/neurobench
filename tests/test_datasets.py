import os

from neurobench.datasets import SpeechCommands
from neurobench.datasets import Gen4DetectionDataLoader
from neurobench.datasets import PrimateReaching
from neurobench.datasets import DVSGesture
from neurobench.datasets import MackeyGlass
from neurobench.datasets import WISDMDataLoader
from torch.utils.data import DataLoader

import torch

dataset_path = "data/"


def test_nehar():
    path = dataset_path + 'nehar/'
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")
    wisdm = WISDMDataLoader(path)
    assert len(wisdm) > 0
    assert list(wisdm.train_dataset[0].shape) == [21720, 40, 6]
    assert list(wisdm.val_dataset[0].shape) == [7240, 40, 6]
    assert list(wisdm.test_dataset[0].shape) == [7241, 40, 6]

    wisdm.setup('fit')
    assert isinstance(wisdm.ds_train, torch.utils.data.TensorDataset)
    assert isinstance(wisdm.ds_val, torch.utils.data.TensorDataset)
    assert isinstance(wisdm.ds_test, torch.utils.data.TensorDataset)
    assert isinstance(wisdm.train_dataloader(), torch.utils.data.DataLoader)
    assert isinstance(wisdm.val_dataloader(), torch.utils.data.DataLoader)
    assert isinstance(wisdm.test_dataloader(), torch.utils.data.DataLoader)

    wisdm = WISDMDataLoader(path)

    wisdm.setup('test')
    assert isinstance(wisdm.ds_test, torch.utils.data.TensorDataset)
    assert wisdm.ds_val is None
    assert wisdm.ds_train is None

    wisdm = WISDMDataLoader(path)
    wisdm.setup('predict')
    assert isinstance(wisdm.ds_test, torch.utils.data.TensorDataset)
    assert wisdm.ds_val is None
    assert wisdm.ds_train is None


def test_speech_commands():
    path = dataset_path + "speech_commands/"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")
    ds = SpeechCommands(path)

    assert len(ds) > 0
    assert ds[0][0].shape
    assert list(ds[0][0].shape) == [16000, 1]  # timesteps, channels
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
    mg = MackeyGlass(17, 0.9)

    assert len(mg) > 0

    trainset = torch.utils.data.Subset(mg, mg.ind_train)
    testset = torch.utils.data.Subset(mg, mg.ind_test)

    assert len(trainset) == mg.traintime_pts
    assert len(testset) == mg.testtime_pts

    assert trainset[0][0].shape == (1, 1)
    assert trainset[0][1].shape == (1,)
    assert testset[0][0].shape == (1, 1)
    assert testset[0][1].shape == (1,)

    assert (torch.eq(trainset[0][0], mg[0][0]))
    assert (torch.eq(trainset[0][1], mg[0][1]))
    assert (torch.eq(testset[0][0], mg[mg.traintime_pts][0]))
    assert (torch.eq(testset[0][1], mg[mg.traintime_pts][1]))


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
        batch_size=1,
        num_tbins=12,
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
    assert data[0].shape == (1, 12, 2, 360, 640)  # batch, timestep, channel, height, width
    assert isinstance(data[1], list)  # list[list[nparr]]
    assert isinstance(data[2], dict)
