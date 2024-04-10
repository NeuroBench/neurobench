from pathlib import Path
from functools import partial
from metavision_ml.data import box_processing as box_api
from metavision_ml.data import SequentialDataLoader
import glob

import numpy as np
import torch

"""
The following files in the precomputed dataset test folder do not have 
corresponding label files, for now they are skipped.
"""
skip_files = [
    "moorea_2019-06-17_test_01_000_1769500000_1829500000.h5",
    "moorea_2019-06-14_000_549500000_609500000.h5",
    "moorea_2019-06-14_000_1220500000_1280500000.h5",
    "moorea_2019-06-26_test_02_000_1708500000_1768500000.h5",
]


def create_class_lookup(wanted_keys=[]):
    """Source code modified from metavision_ml.data.box_processing.create_class_lookup
    to avoid having extraneous label map json file."""
    label_dic = {
        0: "pedestrian",
        1: "two wheeler",
        2: "car",
        3: "truck",
        4: "bus",
        5: "traffic sign",
        6: "traffic light",
    }

    # we take maximum class id + 1 because of class id 0
    size = max(label_dic.keys()) + 1

    # check that all wanted classes are inside the dataset
    classes = label_dic.values()
    if wanted_keys:
        assert any(item != "empty" for item in wanted_keys)
        for key in wanted_keys:
            assert key in classes, "key '{}' not found in the dataset".format(key)
    else:
        # we filter out 'empty' because this is used to annotate empty frames
        wanted_keys = [label for label in classes if label != "empty"]

    wanted_map = {label: idx for idx, label in enumerate(wanted_keys)}
    class_lookup = np.full(size, -1)
    for src_idx in range(size):
        if src_idx not in label_dic:
            continue
        src_label = label_dic[src_idx]
        if src_label not in wanted_keys:
            continue
        class_lookup[src_idx] = wanted_map[src_label] + 1
    return class_lookup


class Gen4DetectionDataLoader(SequentialDataLoader):
    """
    NeuroBench DataLoader for Gen4 pre-computed dataset.

    The default parameters are set for the Gen4 Histograms dataset, which can be
    downloaded from
    https://docs.prophesee.ai/stable/datasets.html#precomputed-datasets
    but you can change that easily by downloading one of the other pre-computed datasets and
    changing the preprocess_function_name and channels parameters accordingly.

    Once downloaded, extract the zip folder and set the dataset_path parameter to the path of the extracted folder.

    """

    def __init__(
        self,
        dataset_path="data/Gen 4 Histograms",
        split="testing",
        batch_size: int = 4,
        num_tbins: int = 12,
        preprocess_function_name="histo",
        delta_t=50000,
        channels=2,  # histograms have two channels
        height=360,
        width=640,
        max_incr_per_pixel=5,
        class_selection=["pedestrian", "two wheeler", "car"],
        num_workers=4,
    ):
        """
        Initializes the Gen4DetectionDataLoader dataloader.

        Args:
            dataset_path: path to the dataset folder
            split: split to use, can be 'training', 'validation' or 'testing'
            batch_size: batch size
            num_tbins: number of time bins in a mini batch
            preprocess_function_name: name of the preprocessing function to use, 'histo' by default. Can be that are listed under https://docs.prophesee.ai/stable/api/python/ml/preprocessing.html
            delta_t: time interval between two consecutive frames
            channels: number of channels in the input data, 2 by default for histograms
            height: height of the input data
            width: width of the input data
            max_incr_per_pixel: maximum number of events per pixel
            class_selection: list of classes to use
            num_workers: number of workers for the dataloader

        """

        self.dataset_path = Path(dataset_path)
        self.files_train = glob.glob(str(self.dataset_path / "train" / "*.h5"))
        self.files_val = glob.glob(str(self.dataset_path / "val" / "*.h5"))
        self.files_test = glob.glob(str(self.dataset_path / "test" / "*.h5"))

        # patch to remove files without labels
        for file in self.files_test:
            if file.split("/")[-1] in skip_files:
                self.files_test.remove(file)

        self.split = split
        self.data = {
            "training": self.files_train,
            "validation": self.files_val,
            "testing": self.files_test,
        }[split]
        self.channels = channels
        self.height = height
        self.width = width
        self.class_selection = class_selection

        # class_lookup = box_api.create_class_lookup(label_map_path, class_selection)
        class_lookup = create_class_lookup(class_selection)

        self.kw_args = dict(
            delta_t=delta_t,
            preprocess_function_name=preprocess_function_name,
            array_dim=[num_tbins, channels, height, width],
            load_labels=partial(
                box_api.load_boxes,
                num_tbins=num_tbins,
                class_lookup=class_lookup,
                min_box_diag_network=60,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            padding=True,
            preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel},
        )
        super().__init__(self.data, **self.kw_args)

    def __next__(self):
        """
        Override the metavision dataloader to reformat data.

        Errata: note that labels do not fit the usual format of tensor with batch as first dimension,
        since the number of boxes per frame is variable.

        Returns:
            inputs (tensor): sample data, shape (batch, timesteps, channels, height, width)
            labels (list[list[bboxes]]): labels of the shape (timesteps, batch, num bounding boxes)
            kwargs (dict): dictionary of metadata with keys 'mask_keep_memory', 'frame_is_labeled', 'video_infos'

        """
        data_dict = super().__next__()

        inputs = data_dict["inputs"]
        inputs = inputs.permute(
            1, 0, 2, 3, 4
        )  # batch, timesteps, channels, height, width

        # list of list of bboxes, shape (timesteps, batch, num bbox)
        # where bbox is structured np array
        labels_bboxes = data_dict["labels"]

        del data_dict["inputs"]
        del data_dict["labels"]

        kwargs = data_dict

        return inputs, labels_bboxes, kwargs
