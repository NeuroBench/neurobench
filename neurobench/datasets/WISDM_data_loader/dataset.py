import numpy as np
import os
import torch


def convert_to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile! :(
            return None
        return directory_path


class DataLoader:
    def __init__(self, file_name):
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val,
         y_test) = self.load_wisdm2_data(file_name)
        self.train_dataset = convert_to_tensor(x_train, np.argmax(y_train, axis=-1))
        self.val_dataset = convert_to_tensor(x_val, np.argmax(y_val, axis=-1))
        self.test_dataset = convert_to_tensor(x_test, np.argmax(y_test, axis=-1))
        self.n_classes = len(np.unique(np.argmax(y_train, axis=-1)))
        self.sample_n_timestamps = len(x_train[0])
        self.sample_n_dim = len(x_train[0][0])

    def get_training_set(self, n_samples=None):
        if n_samples:
            N = self.train_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:n_samples]
            return self.train_dataset[0][ids], self.train_dataset[1][ids]
        return self.train_dataset

    def get_validation_set(self, n_samples=None):
        if n_samples:
            N = self.val_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:n_samples]
            return self.val_dataset[0][ids], self.val_dataset[1][ids]
        return self.val_dataset

    def get_test_set(self, n_samples=None):
        if n_samples:
            N = self.test_dataset[0].shape[0]
            ids = np.array(range(0, N))
            np.random.shuffle(ids)
            ids = ids[:n_samples]
            return self.test_dataset[0][ids], self.test_dataset[1][ids]
        return self.test_dataset

    @staticmethod
    def filter_labels(dataset, labels):
        # return dataset.filter()
        pass

    @staticmethod
    def load_wisdm2_data(file_path):

        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])
