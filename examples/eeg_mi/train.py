import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate

from model import EEG_SNN
from neurobench.datasets import ThorEEGMI


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] Everything seeded with seed={seed}")


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64

train_set = ThorEEGMI(root="../../data", split="train", download=True)
val_set   = ThorEEGMI(root="../../data", split="val",   download=True)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False)


def train_one_epoch(model, loader, optimizer, criterion, n):
    model.train()
    total_loss, correct = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb).sum(1)  # sum spike counts over timesteps: (batch, n_outputs)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
    return total_loss / n, correct / n


def evaluate(model, loader, criterion, n):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb).sum(1)  # sum spike counts over timesteps: (batch, n_outputs)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
    return total_loss / n, correct / n


n_train = len(train_set)
n_val   = len(val_set)

model     = EEG_SNN(n_inputs=train_set.n_channels, n_hidden=256, n_outputs=train_set.n_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 120
best_val_acc     = 0.0
best_model_state = None

print(f"\n{'='*55}")
print(f"  Training EEG-SNN")
print(f"{'='*55}")

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, n_train)
    vl_loss, vl_acc = evaluate(model, val_loader, criterion, n_val)

    if vl_acc > best_val_acc:
        best_val_acc     = vl_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {epoch:03d} | "
        f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.3f} | "
        f"Val loss: {vl_loss:.4f}  acc: {vl_acc:.3f}"
    )

model.load_state_dict(best_model_state)
torch.save(best_model_state, "./model_data/best_model.pt")

print(f"\nBest val acc: {best_val_acc:.4f}")