from pathlib import Path
from functools import partial
from metavision_ml.data import box_processing as box_api
from metavision_ml.data import SequentialDataLoader
import glob


class Gen4DetectionDataLoader(SequentialDataLoader):
    """NeuroBench DataLoader for Gen4 pre-computed dataset

    The default parameters are set for the Gen4 Histograms dataset, which can be downloaded from
    https://docs.prophesee.ai/stable/datasets.html#precomputed-datasets
    but you can change that easily by downloading one of the other pre-computed datasets and
    changing the preprocess_function_name and channels parameters accordingly.

    Once downloaded, extract the zip folder and set the dataset_path parameter to the path of the extracted folder.
    """
    def __init__(
        self,
        dataset_path="data/Gen 4 Histograms",
        split="testing",
        label_map_path="label_map_dictionary.json",
        batch_size: int = 4,
        num_tbins: int = 12,
        preprocess_function_name="histo",
        delta_t=50000,
        channels=2,  # histograms have two channels
        height=360,
        width=640,
        max_incr_per_pixel=5,
        class_selection=["pedestrian", "two wheeler", "car"],
        num_workers=4
    ):
        """ Initializes the Gen4DetectionDataLoader dataloader.

        Args:
            dataset_path: path to the dataset folder
            split: split to use, can be 'training', 'validation' or 'testing'
            label_map_path: path to the label_map_dictionary.json file
            batch_size: batch size
            num_tbins: number of time bins in a mini batch
            preprocess_function_name: name of the preprocessing function to use, 'histo' by default. Can be that are listed under https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/python_api/preprocessing.html#module-metavision_ml.preprocessing.event_to_tensor
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
        class_lookup = box_api.create_class_lookup(label_map_path, class_selection)

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
