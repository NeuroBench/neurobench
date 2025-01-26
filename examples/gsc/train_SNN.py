import torch
import numpy as np
import snntorch.functional as func
import snntorch.surrogate as surrogate
import snntorch.utils as utils

from tqdm import tqdm
from torch.utils.data import DataLoader

from neurobench.datasets import SpeechCommands
from neurobench.preprocessing import S2SPreProcessor
from neurobench.postprocessing import choose_max_count

from SNN import net

BATCH_SIZE = 5
NUM_WORKERS = 8
EPOCHS = 100

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
data_dir = "../../../data/speech_commands/"
train_set = SpeechCommands(path=data_dir, subset="training")
val_set = SpeechCommands(path=data_dir, subset="validation")
test_set = SpeechCommands(path=data_dir, subset="testing")

s2s = S2SPreProcessor()

# Create the dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Different parameters to Speech2Spikes make different shapes of data
tmp_dat = torch.stack([train_set[0][0], train_set[1][0]])
tmp_label = torch.stack([train_set[0][1], train_set[1][1]])
tmp_out = s2s((tmp_dat, tmp_label))
num_steps = tmp_out[0].shape[1]
num_feat = tmp_out[0].shape[2]

# The SNNTorch forward pass
def forward_pass(net, data, num_steps):
    spk_rec = []
    utils.reset(net)
    for step in range(num_steps):
        spk_out, _ = net(data[:, step, ...])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)

# Send network to device
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
loss_fn = func.mse_count_loss(correct_rate=0.25, incorrect_rate=0.025)

# Training Start
best_acc = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}:")
    train_loss = []
    train_acc = []
    net.train()
    for batch in tqdm(iter(train_loader)):
        events, targets = s2s(batch)
        events = events.to(device)
        targets = targets.to(device)

        spk_rec = forward_pass(net, events, num_steps)
        loss_val = loss_fn(spk_rec, targets)

        train_loss.append(loss_val.item())
        train_acc.append(func.accuracy_rate(spk_rec, targets))

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    print(f"Train Loss: {np.mean(train_loss):.3f}")
    print(f"Train Accuracy: {np.mean(train_acc) * 100:.2f}%")

    val_loss = []
    val_acc = []
    net.eval()
    for batch in tqdm(iter(val_loader)):
        events, targets = s2s(batch)
        events = events.to(device)
        targets = targets.to(device)

        spk_rec = forward_pass(net, events, num_steps)

        val_loss.append(loss_fn(spk_rec, targets).item())
        val_acc.append(func.accuracy_rate(spk_rec, targets))

    print(f"Validation Loss: {np.mean(val_loss):.3f}")
    print(f"Validation Accuracy: {np.mean(val_acc) * 100:.2f}%")

    if np.mean(val_acc) > best_acc:
        print("New Best Validation Accuracy. Saving...")
        best_acc = np.mean(val_acc)
        torch.save(net.state_dict(), "model_data/s2s_gsc_snntorch")

    print(f"---------------------\n")

# Load the weights into the network for inference
net.load_state_dict(torch.load("model_data/s2s_gsc_snntorch"))
