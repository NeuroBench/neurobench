"""
Train the NSA HAR LIF baseline.

Hyperparameters default to NSA's published HAR config (HAR/run_all.sh), so a full run
reproduces the NSA baseline. The training loop follows NSA's: AdamW, StepLR (step 10,
gamma 0.8), cross entropy on the time-averaged output, gradient clipping, and the
checkpoint with the best test accuracy is kept.

Usage:
    python train.py                 # feedforward LIF (NSA: --neuron lif)
    python train.py --recurrent     # recurrent LIF  (NSA: --neuron lif --recurrent)
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # examples/nsa, for nsa_common

from data import load_split  # noqa: E402
from model import LIF_FF_CONFIG, LIF_REC_CONFIG, LIFNet  # noqa: E402
from nsa_common import TimeFirstWrapper  # noqa: E402

DEFAULT_DATA = os.path.join(HERE, "..", "..", "..", "data", "nsa", "har")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-path", default=DEFAULT_DATA)
    p.add_argument("--recurrent", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=None, help="NSA: 3e-3 FF, 1.5e-3 rec")
    p.add_argument("--grad-clip", type=float, default=None, help="NSA: 1.0 FF, 0 rec")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-batches", type=int, default=0, help="debug: stop early")
    p.add_argument("--out", default=None, help="checkpoint path")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, grad_clip=0.0, limit=0):
    """One pass over ``loader``. Trains if an optimizer is given. Returns (loss, acc)."""
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if limit and i >= limit:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)  # [B, classes], already averaged over time
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            seen += y.size(0)
    return total_loss / max(seen, 1), 100.0 * correct / max(seen, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    config = dict(LIF_REC_CONFIG if args.recurrent else LIF_FF_CONFIG)
    lr = args.lr if args.lr is not None else (1.5e-3 if args.recurrent else 3e-3)
    grad_clip = args.grad_clip
    if grad_clip is None:
        grad_clip = 0.0 if args.recurrent else 1.0

    name = "har_lif_rec.pt" if args.recurrent else "har_lif_ff.pt"
    out = args.out or os.path.join(HERE, "model_data", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    train_set = load_split(args.data_path, "train")
    test_set = load_split(args.data_path, "test")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(f"train {len(train_set)}  test {len(test_set)}  device {args.device}")

    net = LIFNet(**config)
    model = TimeFirstWrapper(net, readout="mean").to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_acc = -1.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, args.device, optimizer, grad_clip, args.limit_batches
        )
        scheduler.step()
        te_loss, te_acc = run_epoch(
            model, test_loader, args.device, limit=args.limit_batches
        )
        print(
            f"epoch {epoch + 1:3d}  train loss {tr_loss:.4f} acc {tr_acc:.2f}  "
            f"test loss {te_loss:.4f} acc {te_acc:.2f}"
        )
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {"config": config, "state_dict": net.state_dict(), "test_acc": te_acc},
                out,
            )
    print(f"best test acc {best_acc:.2f}, saved to {out}")


if __name__ == "__main__":
    main()
