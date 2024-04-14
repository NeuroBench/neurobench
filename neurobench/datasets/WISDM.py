from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import numpy as np
import torch
from .utils import download_url
from pathlib import Path
from typing import Optional, List, Union
from urllib.error import URLError
from tqdm import tqdm
import zipfile
from sklearn.model_selection import train_test_split

label_map = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "brushing-teeth",
    "H": "eating-soup",
    "I": "eating-chips",
    "J": "eating-pasta",
    "K": "drinking-from-cup",
    "L": "eating-sandwich",
    "M": "kicking(soccer-ball)",
    "O": "playing-catch",
    "P": "dribbling(basketball)",
    "Q": "writing",
    "R": "clapping",
    "S": "folding-clothes",
}


def extract_zip_recursive(zip_path: Union[str, Path], extract_to: Union[str, Path]):
    """
    Recursively extracts zip files including any nested zip files found within. Each
    extracted zip file is then also opened and its contents extracted recursively into a
    new directory based on the zip file's name. Nested zip files are removed after
    extraction.

    Args:
        zip_path (Union[str, Path]): The path to the zip file that needs to be extracted.
                                     Can be a string or a Path object.
        extract_to (Union[str, Path]): The directory where the zip files should be extracted to.
                                       Can be a string or a Path object.

    """

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_files = zip_ref.infolist()
        with tqdm(
            total=len(zip_files), unit="file", desc=f"Extracting {zip_path}"
        ) as progress_bar:
            for file in zip_files:
                zip_ref.extract(file, path=extract_to)
                progress_bar.update(1)
                extracted_file_path = os.path.join(extract_to, file.filename)

                if zipfile.is_zipfile(extracted_file_path):
                    nested_extract_to = os.path.join(
                        extract_to, os.path.splitext(file.filename)[0]
                    )
                    os.makedirs(nested_extract_to, exist_ok=True)
                    extract_zip_recursive(extracted_file_path, nested_extract_to)
                    os.remove(extracted_file_path)


class WISDM(LightningDataModule):
    def __init__(
        self,
        dir_path: Union[str, Path],
        batch_size: int = 256,
        sensor_devices: Optional[List[str]] = None,
        acquisition_devices: Optional[List[str]] = None,
        selected_labels: Optional[List[str]] = None,
        activity_length: int = 40,
    ):
        """
        Initializes the data processing model which includes loading and preparing the
        WISDM dataset for training, validation, and testing.

        Args:
            dir_path (Union[str, Path]): Directory path where the datasets are stored or will be downloaded.
            batch_size (int, optional): The number of samples in each batch of data. Defaults to 256.
            sensor_devices (Optional[List[str]], optional): List of sensor devices to include in the dataset.
                Defaults to ['accel', 'gyro'].
            acquisition_devices (Optional[List[str]], optional): List of acquisition devices to use.
                Defaults to ['watch'].
            selected_labels (Optional[List[str]], optional): List of activity labels to include in the dataset.
                If None, defaults to ['P', 'O', 'F', 'Q', 'R', 'G', 'S'].
            activity_length (int, optional): The number of time steps in each data sample. Defaults to 40.

        For more detailed explanations of the default parameters and their settings, please refer to the
        document `neurobench/examples/nehar/README.md` located in the project's root directory.

        """
        super().__init__()

        self.ds_test = None
        self.ds_val = None
        self.ds_train = None
        self.batch_size = batch_size
        self.root_dir = Path(dir_path)
        self.dataset_folder = (
            self.root_dir if self.root_dir.name == "nehar" else self.root_dir / "nehar"
        )
        self.url = (
            "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and"
            "+biometrics+dataset.zip"
        )

        self.acquisition_devices = (
            ["watch"] if acquisition_devices is None else acquisition_devices
        )
        self.sensor_devices = (
            ["accel", "gyro"] if sensor_devices is None else sensor_devices
        )

        self.selected_labels = (
            selected_labels
            if selected_labels is not None and selected_labels in label_map.keys()
            else ["P", "O", "F", "Q", "R", "G", "S"]
        )
        self.activity_length = activity_length

        self.train_split_ratio = 0.4
        self.val_split_ratio = 0.5

        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm_data()
        self.train_dataset = (
            torch.tensor(x_train, dtype=torch.float),
            torch.tensor(np.argmax(y_train, axis=-1), dtype=torch.uint8),
        )
        self.val_dataset = (
            torch.tensor(x_val, dtype=torch.float),
            torch.tensor(np.argmax(y_val, axis=-1), dtype=torch.uint8),
        )
        self.test_dataset = (
            torch.tensor(x_test, dtype=torch.float),
            torch.tensor(np.argmax(y_test, axis=-1), dtype=torch.uint8),
        )

        self.num_inputs = next(iter(self.train_dataset))[0].shape[1]
        self.num_steps = next(iter(self.train_dataset))[0].shape[0]
        self.num_outputs = len(np.unique(np.argmax(y_train, axis=-1)))

    def download(self):
        """Download the WISDM data if it doesn't exist already."""
        file_path = self.dataset_folder / "wisdm.zip"
        try:
            print(f"Downloading {self.url}")
            download_url(self.url, str(file_path))
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print("Unzipping file...")
            extract_zip_recursive(file_path, self.dataset_folder)
            print()

    def load_wisdm_data(self):
        """
        Load the WISDM data from the dataset folder.

        If the data is not present, download it first.

        """

        self.dataset_folder.mkdir(parents=True, exist_ok=True)
        data_file_path = next(self.dataset_folder.glob("*.npz"), None)
        if data_file_path is None:
            self.download()
            x_samples, y_samples = self.generate_samples()
            x_train, x_val, x_test, y_train, y_val, y_test = self.generate_dataset(
                x_samples, y_samples
            )
            return x_train, x_val, x_test, y_train, y_val, y_test

        data = np.load(data_file_path)
        return (
            data["arr_0"],
            data["arr_1"],
            data["arr_2"],
            data["arr_3"],
            data["arr_4"],
            data["arr_5"],
        )

    def generate_dataset(self, x_samples: np.array, y_samples: np.array):
        """
        Generates a dataset by splitting the input samples into training, validation,
        and test sets.

        The method divides the x and y samples into training, validation, and test subsets based on the
        predefined ratios: `train_split_ratio` and `val_split_ratio`. The splits are used to ensure the
        model is trained, validated, and tested on different subsets of data. The resulting datasets
        are saved to a file in the specified `dataset_folder`.

        Args:
            x_samples (np.array): The input features array. Should be a NumPy array where each row corresponds to an example.
            y_samples (np.array): The target values array. Should be a NumPy array that aligns with `x_samples`.

        Returns:
            tuple: A tuple containing six elements:
                - x_train (np.array): The training set features.
                - x_val (np.array): The validation set features.
                - x_test (np.array): The test set features.
                - y_train (np.array): The training set labels.
                - y_val (np.array): The validation set labels.
                - y_test (np.array): The test set labels.

        """

        x_train, x_valtest, y_train, y_valtest = train_test_split(
            x_samples, y_samples, test_size=self.train_split_ratio, random_state=42
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_valtest, y_valtest, test_size=self.val_split_ratio, random_state=42
        )

        file_name = "data_watch" + "_subset" + "_" + str(40)
        filepath = self.dataset_folder / f"{file_name}.npz"

        np.savez(filepath, x_train, x_val, x_test, y_train, y_val, y_test)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def generate_samples(self):
        """
        Generates sample subsets from sensor data. The method first creates a
        consolidated DataFrame using the `generate_dataframe` function. It then
        processes this data to generate fixed- size subsets for each activity label,
        each representing a fixed-size input and its corresponding label.

        The data for each subset is filtered by subject ID and activity label, ensuring that
        each subset homogeneously represents the same type of activity. The subsets are generated
        to be precisely the length specified by `self.activity_length`.

        Returns:
            tuple: A tuple (x_samples, y_samples) where:
                - x_samples (np.array): An array of data samples, each corresponding to a fixed-size
                  window of sensor data for a specific activity.
                - y_samples (np.array): A one-hot encoded array of labels corresponding to the activities
                  represented by `x_samples`.

        """

        wisdm_dataframe = self.generate_dataframe()

        def process_subject_label(subject_id, label_index, label):
            filtered_df = wisdm_dataframe[
                (wisdm_dataframe["Subject-id"] == subject_id)
                & (wisdm_dataframe["Activity Label"] == label)
            ].drop(["Subject-id", "Activity Label", "Timestamp"], axis=1)

            # Create subsets and return them along with their corresponding label index
            return [
                (
                    filtered_df.iloc[start : start + self.activity_length].values,
                    label_index,
                )
                for start in range(0, len(filtered_df), self.activity_length)
                if len(filtered_df.iloc[start : start + self.activity_length])
                == self.activity_length
            ]

        samples = [
            process_subject_label(subject_id, label_index, label)
            for subject_id in range(1600, 1651)
            for label_index, label in enumerate(self.selected_labels)
        ]
        x_samples, y_samples = zip(
            *[sample for sublist in samples for sample in sublist if len(sublist) > 0]
        )

        return (
            np.array(x_samples),
            np.eye(len(self.selected_labels), dtype="uint8")[np.array(y_samples)],
        )

    def generate_dataframe(self):
        """
        Generates a consolidated DataFrame from multiple acquisition devices and sensor
        data, optionally filtering by specific activity labels. The function processes
        data from different sensor devices and acquisition devices, merges them on
        common columns, and aligns them by timestamps to create a unified dataset.

        Returns:
            pandas.DataFrame: A consolidated DataFrame containing merged data from the specified
            sensor and acquisition devices, filtered by the selected activity labels.

        Note:
            - The function assumes that the data for each sensor and acquisition device can be loaded
              using the `load_data_from_csv` function.
            - Data is merged based on 'Subject-id', 'Activity Label', and 'Timestamp' columns.
            - If data from only one sensor or acquisition device is provided, the function returns
              the data from that sensor or device without merging.
            - The function aligns data on a per-subject and per-activity label basis, ensuring that
              the timestamps match across different sensors and devices.

        """

        devices_data = {}

        for acquisition_device in self.acquisition_devices:
            sensors_data = {
                sensor: self.load_data_from_csv(sensor, acquisition_device)
                for sensor in self.sensor_devices
            }

            if len(sensors_data) == 2:
                common_columns = ["Subject-id", "Activity Label", "Timestamp"]
                sorted_data = {
                    key: data.sort_values(by=common_columns)
                    for key, data in sensors_data.items()
                }

                # Consolidate the data merging and concatenation into a single step
                devices_data[acquisition_device] = pd.concat(
                    [
                        pd.merge_asof(
                            sorted_data[self.sensor_devices[0]].query(
                                "`Subject-id` == @subject_id & `Activity Label` == @label"
                            ),
                            sorted_data[self.sensor_devices[1]].query(
                                "`Subject-id` == @subject_id & `Activity Label` == @label"
                            ),
                            on="Timestamp",
                            by=["Subject-id", "Activity Label"],
                        )
                        for subject_id in range(1600, 1651)
                        for label in self.selected_labels
                    ],
                    ignore_index=True,
                ).dropna()
            else:
                devices_data[acquisition_device] = next(iter(sensors_data.values()))

        if len(devices_data) == 2:
            for device in self.acquisition_devices:
                data = devices_data[device].sort_values(
                    by=["Subject-id", "Activity Label", "Timestamp"]
                )
                concatenated_data = [
                    data[
                        (data["Subject-id"] == sid) & (data["Activity Label"] == label)
                    ]
                    .reset_index(drop=True)
                    .assign(Timestamp=lambda x: x.index)
                    for sid in range(1600, 1651)
                    for label in self.selected_labels
                ]
                devices_data[device] = pd.concat(concatenated_data, ignore_index=True)

            merged_devices_data = pd.merge(
                devices_data[self.acquisition_devices[0]],
                devices_data[self.acquisition_devices[1]],
                on=["Subject-id", "Activity Label", "Timestamp"],
                copy=False,
            ).dropna()
            return merged_devices_data
        else:
            return next(iter(devices_data.values()))

    def load_data_from_csv(self, sensor_device: str, acquisition_device: str):
        """
        Loads and processes data from CSV files for a given sensor and acquisition
        device from a specified directory, combining all CSV files into a single
        DataFrame, and resampling the data.

        The function reads all text files from the specified directory, which is constructed using the
        `sensor_device` and `acquisition_device` parameter. The data is then combined into a single
        pandas DataFrame, with timestamps converted to datetime objects. Each group of data, categorized by
        'Subject-id' and 'Activity Label', is resampled at a 50ms rate due to some issues highlighted in
        https://arxiv.org/abs/2305.10222.

        Args:
            sensor_device (str): The name of the sensor device (e.g., 'accel').
            acquisition_device (str): The name of the acquisition device (e.g., 'watch').

        Returns:
            pandas.DataFrame: A DataFrame containing the resampled data, with columns for
            'Subject-id', 'Activity Label', 'Timestamp', and the x, y, z values for the
            specified sensor and acquisition device.

        Note:
            - The function expects data files to be in text format with a predefined set of column names.
            - Data is resampled based on a fixed 50ms rate, and any missing data after resampling is dropped.
            - The directory structure is expected to follow the pattern
              '{self.dataset_folder}/wisdm-dataset/wisdm-dataset/raw/{acquisition_device}/{sensor_device}'.

        """
        path = (
            self.dataset_folder
            / f"wisdm-dataset/wisdm-dataset/raw/{acquisition_device}/{sensor_device}"
        )
        columns = [
            "Subject-id",
            "Activity Label",
            "Timestamp",
            f"{acquisition_device}_{sensor_device}_x",
            f"{acquisition_device}_{sensor_device}_y",
            f"{acquisition_device}_{sensor_device}_z",
        ]

        def remove_semicolon(s):
            return s.str.rstrip(";")

        data = pd.concat(
            [
                pd.read_csv(file, header=None, names=columns, sep=",").apply(
                    lambda col: (
                        remove_semicolon(col) if col.name == columns[-1] else col
                    )
                )
                for file in path.rglob("*.txt")
            ]
        )

        data[columns[-3:]] = data[columns[-3:]].astype(float)

        data["Timestamp"] = pd.to_datetime(data["Timestamp"])

        def resample(group):
            group.set_index("Timestamp", inplace=True)
            return group.resample("50ms").first().dropna().reset_index()

        grouped_data = (
            data.groupby(["Subject-id", "Activity Label"])
            .apply(resample)
            .reset_index(drop=True)
        )
        return grouped_data

    def setup(self, stage: str):
        """Prepares datasets for different stages of the model training process."""
        match stage:
            case "fit":
                x_train, y_train = self.train_dataset
                x_val, y_val = self.val_dataset
                x_test, y_test = self.test_dataset
                self.ds_train = TensorDataset(x_train, y_train)
                self.ds_val = TensorDataset(x_val, y_val)
                self.ds_test = TensorDataset(x_test, y_test)

            case "test":
                x_test, y_test = self.test_dataset
                self.ds_test = TensorDataset(x_test, y_test)

            case "predict":
                x_test, y_test = self.test_dataset
                self.ds_test = TensorDataset(x_test, y_test)

    def train_dataloader(self):
        """Constructs a DataLoader for training."""
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Constructs a DataLoader for validation."""
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Constructs a DataLoader for testing."""
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        """Constructs a DataLoader for prediction."""
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def __len__(self):
        return (
            self.train_dataset[0].shape[0]
            + self.val_dataset[0].shape[0]
            + self.test_dataset[0].shape[0]
        )
