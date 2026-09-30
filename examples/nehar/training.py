import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from neurobench.datasets import WISDM
from SCNN import SCNN

def run_epoch(model, loader, loss_fn, accuracy_metric, optimizer, device, train: bool):
    model.train(train)
    accuracy_metric.reset()
    total_loss    = 0.0
    total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            spk_output = model.forward(data)
            logits = spk_output.sum(1)
            loss   = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy_metric.update(logits, labels)
            total_loss    += loss.item() * len(labels)
            total_samples += len(labels)

    mean_loss = total_loss / total_samples
    accuracy  = accuracy_metric.compute().item()
    return mean_loss, accuracy

if __name__ == "__main__":
    
    BATCH_SIZE   = 256
    LR           = 1e-3
    MAX_EPOCHS   = 100
    DATASET_ROOT = "../../data/nehar"
    CKPT_PATH    = "./model_data/WISDM_snnTorch.pt"
    NUM_WORKERS  = min(8, os.cpu_count() or 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = WISDM(root=DATASET_ROOT, split="train", download=True)
    val_set   = WISDM(root=DATASET_ROOT, split="val",   download=True)
    test_set  = WISDM(root=DATASET_ROOT, split="test",  download=True)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=False, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False, persistent_workers=True,
    )

    print(train_set)
    print(val_set)
    print(test_set)

    model     = SCNN().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    n_classes       = train_set.n_classes
    accuracy_metric = MulticlassAccuracy(num_classes=n_classes, average="macro").to(device)

    best_val_acc = 0.0

    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, accuracy_metric, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, loss_fn, accuracy_metric, optimizer=None, device=device, train=False
        )

        print(
            f"Epoch {epoch:>3}/{MAX_EPOCHS} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                CKPT_PATH,
            )

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.4f})")

    test_loss, test_acc = run_epoch(
        model, test_loader, loss_fn, accuracy_metric, optimizer=None, device=device, train=False
    )
    print(f"\nTest results | loss {test_loss:.4f}  macro-accuracy {test_acc:.4f}")