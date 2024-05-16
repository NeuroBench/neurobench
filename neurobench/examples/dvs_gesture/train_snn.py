import snntorch as snn
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set the random seed for PyTorch
def rand_seed(n):
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(n)

lr = 0.008273059787948487
batch_size = 64
train_time_bin = 25
test_time_bin = 150
data_dir = './data'

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Neuron Hyperparameters
        beta_1 = 0.9999903192467171
        beta_2 = 0.7291118090686332
        beta_3 = 0.9364650136740154
        beta_4 = 0.8348241794080301
        threshold_1 = 3.511291184386264
        threshold_2 = 3.494437965584431
        threshold_3 = 1.5986853560315544
        threshold_4 = 0.3641469130041378
        spike_grad = surrogate.atan()
        
         # Initialize layers
        self.conv1 = nn.Conv2d(2, 16, 5, padding="same")
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta_1, threshold=threshold_1, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(16, 32, 5, padding="same")
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta_2, threshold=threshold_2, spike_grad=spike_grad)
        
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.pool3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta_3, threshold=threshold_3, spike_grad=spike_grad)
        
        self.linear1 = nn.Linear(64*4*4, 11)
        self.dropout_4 = nn.Dropout(0.5956071342984011)
        self.lif4 = snn.Leaky(beta=beta_4, threshold=threshold_4, spike_grad=spike_grad)

    def forward(self, x, quanting=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # Record the final layer
        spk4_rec = []
        mem4_rec = []

        for step in range(x.size(0)):
            # Layer 1
            y = self.conv1(x[step])
            y = self.pool1(y)
            spk1, mem1 = self.lif1(y, mem1)

            # Layer 2
            y = self.conv2(spk1)
            y = self.pool2(y)
            spk2, mem2 = self.lif2(y, mem2)

            # Layer 3
            y = self.conv3(spk2)
            y = self.pool3(y)
            spk3, mem3 = self.lif3(y, mem3)

            # Layer 4
            y = self.linear1(spk3.flatten(1))
            y = self.dropout_4(y)
            spk4, mem4 = self.lif4(y, mem4)

            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        return torch.stack(spk4_rec, dim=0), torch.stack(mem4_rec, dim=0)
            
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
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
    test_loader = DataLoader(cached_testset, batch_size=batch_size,
                             collate_fn=tonic.collation.PadTensors(batch_first=False))

    return train_loader, test_loader

def calc_accuracy(spikes, labels):
    _, idx = spikes.sum(dim=0).max(1)
    batch_acc = (labels == idx).sum()
    return batch_acc

def batch_accuracy(loader, net):
    # print("Accuracy test")
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        loss = 0

        running_length = 0
        running_total = 0

        count = 0

        loss_ce = SF.mse_count_loss()

        loader = iter(loader)
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = net(data)

            batch_acc = calc_accuracy(spk_rec, targets)
            running_length += len(targets)
            running_total += batch_acc
            acc = running_total/running_length
            total += spk_rec.size(1)
            loss += loss_ce(spk_rec, targets)
            count += 1

    return acc, (loss / count)  #(acc / total) , (loss / count)

if __name__ == '__main__':

    rand_seed(1234)

    train_loader, test_loader = dataloader()

    net = Net().to(device)
    num_epochs = 300

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_fn = SF.mse_count_loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8250, eta_min=0, last_epoch=-1)

    max_acc = 0
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for i, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec, mem_rec = net(data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            test_acc, test_loss = batch_accuracy(test_loader, net)

            acc = "{:.4f}".format(test_acc.item())

            if test_acc > max_acc:
                print("New Max Accuracy Found", test_acc.item())
                max_acc = test_acc
                torch.save(net.state_dict(), "./model_data/epoch_" + str(epoch) + "_acc_" + str(acc) + ".pth")
