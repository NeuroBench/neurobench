import pytorch_lightning as pl
from .dataset import DataLoader as DataLoad
from torch.utils.data import DataLoader, TensorDataset


class WISDMDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 256):
        super().__init__()
        self.ds_test = None
        self.ds_val = None
        self.ds_train = None
        self.data_set = DataLoad(data_dir)
        self.batch_size = batch_size
        self.num_inputs = next(iter(self.data_set.train_dataset))[0].shape[1]
        self.num_steps = next(iter(self.data_set.train_dataset))[0].shape[0]
        self.num_outputs = self.data_set.n_classes

    def setup(self, stage: str):
        match stage:
            case 'fit':
                x_train, y_train = self.data_set.train_dataset
                x_val, y_val = self.data_set.val_dataset
                x_test, y_test = self.data_set.test_dataset
                self.ds_train = TensorDataset(x_train, y_train)
                self.ds_val = TensorDataset(x_val, y_val)
                self.ds_test = TensorDataset(x_test, y_test)

            case 'test':
                x_test, y_test = self.data_set.test_dataset
                self.ds_test = TensorDataset(x_test, y_test)

            case 'predict':
                x_test, y_test = self.data_set.test_dataset
                self.ds_test = TensorDataset(x_test, y_test)
                self.predict_dataset = DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=8, shuffle=False, drop_last=False)

    def predict_dataloader(self):
        return self.predict_dataset

    def teardown(self, stage: str):
        ...
