import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import snntorch.utils as utils

import numpy as np

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

from snn import Net

from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set the random seed for PyTorch
def rand_seed(n):
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(n)

# The SNNTorch forward pass
def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)
    for step in range(data.shape[1]):
        spk_out, _ = net(data[:, step, ...])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)

lr = 0.008273059787948487
batch_size = 64
train_time_bin = 25
test_time_bin = 150
epochs = 100
data_dir = './data'
            
def dataloader():
    # sensor_size = tonic.datasets.DVSGesture.sensor_size
    sensor_size = (32, 32, 2)

    train_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                          transforms.Downsample(spatial_factor=0.25),
                                          transforms.ToFrame(sensor_size=sensor_size,
                                                             n_time_bins=train_time_bin),
                                          ])

    test_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                         transforms.Downsample(spatial_factor=0.25),
                                         transforms.ToFrame(sensor_size=sensor_size,
                                                            n_time_bins=test_time_bin),
                                         ])

    trainset = tonic.datasets.DVSGesture(save_to=data_dir, transform=train_transform, train=True)
    testset = tonic.datasets.DVSGesture(save_to=data_dir, transform=test_transform, train=False)

    cached_trainset = DiskCachedDataset(trainset, cache_path='./data/cache/dvs/train')
    cached_testset = DiskCachedDataset(testset, cache_path='./data/cache/dvs/test')

    train_loader = DataLoader(cached_trainset, batch_size=batch_size,
                              collate_fn=tonic.collation.PadTensors(batch_first=True))
    # test whole validation set at once so that accuracy is exact
    test_loader = DataLoader(cached_testset, batch_size=512,
                             collate_fn=tonic.collation.PadTensors(batch_first=True))

    return train_loader, test_loader

if __name__ == '__main__':

    rand_seed(1234)

    train_loader, test_loader = dataloader()

    net = Net().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_fn = SF.mse_count_loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8250, eta_min=0, last_epoch=-1)

    # Training Start
    best_acc = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        train_loss = []
        train_acc = []
        net.train()
        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)

            train_loss.append(loss_val.item())
            train_acc.append(SF.accuracy_rate(spk_rec, targets))

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            scheduler.step()

        print(f"Train Loss: {np.mean(train_loss):.3f}")
        print(f"Train Accuracy: {np.mean(train_acc) * 100:.2f}%")

        val_loss = []
        val_acc = []
        net.eval()
        for data, targets in tqdm(iter(test_loader)):
            data = data.to(device)
            targets = targets.to(device)

            spk_rec = forward_pass(net, data)

            val_loss.append(loss_fn(spk_rec, targets).item())
            val_acc.append(SF.accuracy_rate(spk_rec, targets))

        print(f"Test Loss: {np.mean(val_loss):.3f}")
        print(f"Test Accuracy: {np.mean(val_acc) * 100:.2f}%")

        if np.mean(val_acc) > best_acc:
            print("New Best Test Accuracy. Saving...")
            best_acc = np.mean(val_acc)
            torch.save(net.state_dict(), "model_data/dvs_gesture_snn")

        print(f"---------------------\n")

    # Load the weights into the network for inference and benchmarking
    net.load_state_dict(torch.load("model_data/dvs_gesture_snn"))
