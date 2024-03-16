from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import torch
from .utils import download_url


class WISDM(LightningDataModule):
    """
        Subset version (https://github.com/neuromorphic-polito/NeHAR/blob/main/data/data_watch_subset2_40.npz)
        of the original WISDM dataset (https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
        for a classification task in human activity recognition.

        This data subset, employed for this task, centers on the information gathered from the smartwatch.
        The dataset comprises 7 classes and 36,201 samples, which are split into training, validation, and test sets with proportions of 60%, 20%, and 20%, respectively.

    """

    def __init__(self, path: str = "./data_watch_subset2_40.npz", batch_size: int = 256):
        """
            Initialize the class with the path to the dataset file and the batch size for processing.

        Args:
            path (str): The path to the directory storing the dataset file.
            batch_size (int): The size of the data batches to be used for processing.
        """
        super().__init__()
        self.ds_test = None
        self.ds_val = None
        self.ds_train = None
        self.batch_size = batch_size

        self.url = 'https://media.githubusercontent.com/media/neuromorphic-polito/NeHAR/main/data/data_watch_subset2_40.npz'

        (x_train, x_val, x_test, y_train, y_val,
         y_test) = self.load_wisdm_data(path)
        self.train_dataset = x_train, np.argmax(y_train, axis=-1)
        self.val_dataset = x_val, np.argmax(y_val, axis=-1)
        self.test_dataset = x_test, np.argmax(y_test, axis=-1)

        self.num_inputs = next(iter(self.train_dataset))[0].shape[1]
        self.num_steps = next(iter(self.train_dataset))[0].shape[0]
        self.num_outputs = len(np.unique(np.argmax(y_train, axis=-1)))

    def load_wisdm_data(self, path: str):
        """
            Load the WISDM dataset, downloading it if not present, and return as PyTorch tensors.

            Args:
                path (str): Path to the dataset or directory to save the dataset.

            Returns:
                tuple: A tuple of PyTorch tensors representing the dataset.
            """
        if path.endswith(".npz"):
            file_path = path
            dir_path = os.path.split(file_path)[0]
        else:
            file_path = os.path.join(path, 'data_watch_subset2_40.npz')
            dir_path = path

        if not os.path.exists(dir_path) or not os.path.isfile(file_path):
            print("downloading ....")
            os.makedirs(dir_path, exist_ok=True)
            download_url(self.url, path)

        data = np.load(file_path)
        return tuple(torch.tensor(data[key], dtype=torch.float) for key in data)

    def setup(self, stage: str):
        match stage:
            case 'fit':
                self.ds_train = TensorDataset(*self.train_dataset)
                self.ds_val = TensorDataset(*self.val_dataset)
                self.ds_test = TensorDataset(*self.test_dataset)

            case 'test':
                self.ds_test = TensorDataset(*self.test_dataset)

            case 'predict':
                self.ds_test = TensorDataset(*self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False,
                          persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False,
                          persistent_workers=True)

    def __len__(self):
        return self.train_dataset[0].shape[0] + self.val_dataset[0].shape[0] + self.test_dataset[0].shape[0]
