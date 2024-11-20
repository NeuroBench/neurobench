import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchaudio

import torch.nn.functional as F


from neurobench.datasets import SpeechCommands

from ANN import M5

BATCH_SIZE = 256
NUM_WORKERS = 8
EPOCHS = 50

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
data_dir = "../../../data/speech_commands/"
train_set = SpeechCommands(path=data_dir, subset="training")
val_set = SpeechCommands(path=data_dir, subset="validation")
test_set = SpeechCommands(path=data_dir, subset="testing")

# Create the dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = M5()
model.to(device)

transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.permute(0, 2, 1).to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def validate(model, epoch):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data = data.permute(0, 2, 1).to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

    return correct / len(val_loader.dataset)

# Training Start
best_acc = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}:")
    model.train()
    train(model, epoch, log_interval=20)

    val_acc = []
    # validate
    val_acc.append(validate(model, epoch))

    print(f"Validation Accuracy: {np.mean(val_acc) * 100:.2f}%")

    if np.mean(val_acc) > best_acc:
        print("New Best Validation Accuracy. Saving...")
        best_acc = np.mean(val_acc)
        torch.save(model.state_dict(), "model_data/m5_ann")

    scheduler.step()

    print(f"---------------------\n")

# Load the weights into the network for inference
model.load_state_dict(torch.load("model_data/m5_ann"))