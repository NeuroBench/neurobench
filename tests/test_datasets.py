import os

from neurobench.datasets import SpeechCommands
from neurobench.datasets import Gen4DetectionDataLoader
from neurobench.datasets import PrimateReaching
from neurobench.datasets import MackeyGlass
from neurobench.datasets import WISDM
from neurobench.datasets import ThorEEGMI

import torch

dataset_path = "data/"


def test_eeg_mi():
    path = os.path.join(dataset_path, "eeg_mi")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Can't find {path}")

    train_set = ThorEEGMI(root=path, split="train", download=False)
    assert len(train_set) > 0
    assert list(train_set.data.shape) == [7344, 250, 62]
    assert list(train_set.targets.shape) == [7344]
    assert train_set.n_timesteps == 250
    assert train_set.n_channels == 62

    val_set = ThorEEGMI(root=path, split="val", download=False)
    assert len(val_set) > 0
    assert list(val_set.data.shape) == [1728, 250, 62]
    assert list(val_set.targets.shape) == [1728]

    sample, label = train_set[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert list(sample.shape) == [250, 62]
    assert label.ndim == 0

    samples, labels = train_set[[0, 1, 2]]
    assert list(samples.shape) == [3, 250, 62]
    assert list(labels.shape) == [3]


def test_nehar():
    path = os.path.join(dataset_path, "nehar")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Can't find {path}")

    train_set = WISDM(root=path, split="train", download=False)
    assert len(train_set) > 0
    assert list(train_set.data.shape) == [18907, 40, 6]
    assert list(train_set.targets.shape) == [18907]
    assert train_set.n_timesteps == 40
    assert train_set.n_channels == 6

    val_set = WISDM(root=path, split="val", download=False)
    assert len(val_set) > 0
    assert list(val_set.data.shape) == [6302, 40, 6]
    assert list(val_set.targets.shape) == [6302]

    test_set = WISDM(root=path, split="test", download=False)
    assert len(test_set) > 0
    assert list(test_set.data.shape) == [6303, 40, 6]
    assert list(test_set.targets.shape) == [6303]

    sample, label = train_set[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert list(sample.shape) == [40, 6]
    assert label.ndim == 0

    samples, labels = train_set[[0, 1, 2]]
    assert list(samples.shape) == [3, 40, 6]
    assert list(labels.shape) == [3]

    assert train_set.n_classes == test_set.n_classes


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


def test_mackey_glass():
    filepath = dataset_path + "mackey_glass/mg_17.npy"
    try:
        assert os.path.exists(filepath)
    except AssertionError:
        raise FileExistsError(f"Can't find {filepath}")
    mg = MackeyGlass(file_path=filepath)

    assert len(mg) > 0

    trainset = torch.utils.data.Subset(mg, mg.ind_train)
    testset = torch.utils.data.Subset(mg, mg.ind_test)

    assert len(trainset) == mg.traintime_pts
    assert len(testset) == mg.testtime_pts

    assert trainset[0][0].shape == (1, 1)
    assert trainset[0][1].shape == (1,)
    assert testset[0][0].shape == (1, 1)
    assert testset[0][1].shape == (1,)

    assert torch.eq(trainset[0][0], mg[0][0])
    assert torch.eq(trainset[0][1], mg[0][1])
    assert torch.eq(testset[0][0], mg[mg.traintime_pts][0])
    assert torch.eq(testset[0][1], mg[mg.traintime_pts][1])

    bin_window = 3
    mg = MackeyGlass(file_path=filepath, bin_window=bin_window)
    trainset = torch.utils.data.Subset(mg, mg.ind_train)
    testset = torch.utils.data.Subset(mg, mg.ind_test)

    assert trainset[0][0].shape == (bin_window, 1)
    assert trainset[0][1].shape == (1,)
    assert testset[0][0].shape == (bin_window, 1)
    assert testset[0][1].shape == (1,)

    assert torch.eq(
        mg[0][1], mg[1][0][-1]
    )  # ensure target from previous timestep is appended to lookback window


def test_1mp():
    path = dataset_path + "Gen 4 Histograms/"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")

    dl = Gen4DetectionDataLoader(
        dataset_path=path,
        split="testing",
        batch_size=1,
        num_tbins=12,
        preprocess_function_name="histo",
        delta_t=50000,
        channels=2,  # histograms have two channels
        height=360,
        width=640,
        max_incr_per_pixel=5,
        class_selection=["pedestrian", "two wheeler", "car"],
        num_workers=1,
    )

    assert len(dl) > 0
    assert iter(dl) is not None
    assert next(iter(dl)) is not None

    data = next(iter(dl))
    assert len(data) == 3
    assert data[0].shape == (
        1,
        12,
        2,
        360,
        640,
    )  # batch, timestep, channel, height, width
    assert isinstance(data[1], list)  # list[list[nparr]]
    assert isinstance(data[2], dict)


def test_primate_reaching():
    path = dataset_path + "primate_reaching/PrimateReachingDataset"
    try:
        assert os.path.exists(path)
    except AssertionError:
        raise FileExistsError(f"Can't find {path}")

    filename = "indy_20170131_02.mat"
    file_path = os.path.join(path, filename)
    try:
        assert not os.path.exists(file_path)
    except AssertionError:
        print(f"Dataset {filename} already exists in {path}, not downloading")

    dataset = PrimateReaching(
        file_path=path,
        filename="indy_20170131_02.mat",
        num_steps=250,
        train_ratio=0.8,
        bin_width=0.004,
        biological_delay=50,
        download=True,
    )

    # check dataset non-empty
    assert len(dataset)

    train_set_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, dataset.ind_train), batch_size=1, shuffle=False
    )
    test_set_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, dataset.ind_test), batch_size=1, shuffle=False
    )

    # correct amount of samples
    assert len(train_set_loader) == len(dataset.ind_train)
    assert len(test_set_loader) == len(dataset.ind_test)

    # correct shapes
    assert next(iter(train_set_loader))[0].shape == (1, 250, 96)
    assert next(iter(train_set_loader))[1].shape == (1, 2)
    assert next(iter(test_set_loader))[0].shape == (1, 250, 96)
    assert next(iter(test_set_loader))[1].shape == (1, 2)
