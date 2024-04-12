from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
import os
import gdown
import numpy as np
import torch


def convert_to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)


class WISDM(LightningDataModule):
    def __init__(self, path: str = "path/to/file", batch_size: int = 256):
        super().__init__()
        self.ds_test = None
        self.ds_val = None
        self.ds_train = None
        self.batch_size = batch_size

        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(path)
        self.train_dataset = convert_to_tensor(x_train, np.argmax(y_train, axis=-1))
        self.val_dataset = convert_to_tensor(x_val, np.argmax(y_val, axis=-1))
        self.test_dataset = convert_to_tensor(x_test, np.argmax(y_test, axis=-1))

        self.num_inputs = next(iter(self.train_dataset))[0].shape[1]
        self.num_steps = next(iter(self.train_dataset))[0].shape[0]
        self.num_outputs = len(np.unique(np.argmax(y_train, axis=-1)))

    @staticmethod
    def load_wisdm2_data(path):
        if path.endswith(".npz"):
            file_path = path
            dir_path = os.path.split(file_path)[0]
        else:
            file_path = os.path.join(path, "watch_subset2_40.npz")
            dir_path = path

        if not os.path.exists(dir_path) or not os.path.isfile(file_path):
            os.makedirs(dir_path, exist_ok=True)
            url = "https://drive.google.com/drive/folders/1WCN-XwLM_D2nOTZLY00iGwEJLwDQaUCv"
            gdown.download_folder(url, quiet=True, use_cookies=False, output=dir_path)

        data = np.load(file_path)
        return (
            data["arr_0"],
            data["arr_1"],
            data["arr_2"],
            data["arr_3"],
            data["arr_4"],
            data["arr_5"],
        )

    def setup(self, stage: str):
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
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):
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
