from torch.utils.data import DataLoader

def data_loader(dataset_train, dataset_val, dataset_test, batch_size):

    train_loader = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, drop_last=True, shuffle=True)
    return train_loader, val_loader, test_loader
