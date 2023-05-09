import torch
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics

import spikingjelly

class ClassificationLitModule(pl.LightningModule):

    def __init__(self, model, epochs=10, lr=5e-3, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        self.lr, self.epochs = lr, epochs
        self.num_classes = num_classes
        self.all_nnz, self.all_nnumel = 0, 0

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.train_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.val_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.test_acc_by_class = torchmetrics.Accuracy(num_classes=num_classes, average="none")
        self.train_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x).sum(dim=1)

    def step(self, batch, batch_idx, mode):
        events, target = batch

        outputs = self(events)
        loss = nn.functional.cross_entropy(outputs, target)

        # Measure sparsity if testing
        if mode=="test":
            self.process_nz(self.model.get_nz_numel())

        # Metrics computation
        sm_outputs = outputs.softmax(dim=-1)

        acc, acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        confmat = getattr(self, f'{mode}_confmat')

        acc(sm_outputs, target)
        acc_by_class(sm_outputs, target)
        confmat(sm_outputs, target)

        if mode != "test":
            self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=(mode == "train"))

        spikingjelly.clock_driven.functional.reset_net(self.model)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")

    def on_test_epoch_start(self):
        self.model.add_hooks()

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")

    def on_mode_epoch_end(self, mode):
        print()

        mode_acc, mode_acc_by_class = getattr(self, f"{mode}_acc"), getattr(self, f"{mode}_acc_by_class")
        acc, acc_by_class = mode_acc.compute(), mode_acc_by_class.compute()
        for i,acc_i in enumerate(acc_by_class):
            self.log(f'{mode}_acc_{i}', acc_i)
        self.log(f'{mode}_acc', acc)

        print(f"{mode} accuracy: {100*acc:.2f}%")
        print(f"Cars {100*acc_by_class[0]:.2f}% - Background {100*acc_by_class[1]:.2f}%")
        mode_acc.reset()
        mode_acc_by_class.reset()

        print(f"{mode} confusion matrix:")
        self_confmat = getattr(self, f"{mode}_confmat")
        confmat = self_confmat.compute()
        self.log(f'{mode}_confmat', confmat)
        print(confmat)
        self_confmat.reset()

        if mode=="test":
            print(f"Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0

    def on_train_epoch_end(self):
        self.on_mode_epoch_end(mode="train")

    def on_validation_epoch_end(self):
        self.on_mode_epoch_end(mode="val")

    def on_test_epoch_end(self):
        self.on_mode_epoch_end(mode="test")

    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            if "act" in module:
                nnumel = numel[module]
                if nnumel != 0:
                    total_nnz += nnz
                    total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}