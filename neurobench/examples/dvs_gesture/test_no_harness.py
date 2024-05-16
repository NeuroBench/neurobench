import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate

import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

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

net = Net()
net.load_state_dict(torch.load("model_data/dvs_gesture_snn"))

net.eval()

# Load the dataset
data_dir = "../../../data/dvs_gesture" # data in repo root dir

test_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                         transforms.Downsample(spatial_factor=0.25),
                                         transforms.ToFrame(sensor_size=(32, 32, 2),
                                                            n_time_bins=150),
                                         ])

testset = tonic.datasets.DVSGesture(save_to=data_dir, transform=test_transform, train=False)

test_loader = DataLoader(testset, batch_size=16,
                             collate_fn=tonic.collation.PadTensors(batch_first=False))

def calc_accuracy(spikes, labels):
    _, idx = spikes.sum(dim=0).max(1)
    batch_acc = (labels == idx).sum()
    return batch_acc

total_correct = 0
total = 0

for data, targets in tqdm(test_loader):
	spk_rec, _ = net(data)
	batch_correct = calc_accuracy(spk_rec, targets)
	total_correct += batch_correct
	total += spk_rec.size(1)

print(f"Accuracy: {total_correct / total * 100:.2f}%")

breakpoint()