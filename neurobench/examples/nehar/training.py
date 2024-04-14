from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from neurobench.datasets import WISDM
from SCNN import SCNN
import torch.nn as nn
import torch


class SpikingNetwork(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = SCNN()
        self.loss_fn = nn.CrossEntropyLoss()
        self.running_length = 0
        self.running_total = 0
        self.lr = lr
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer

    def training_step(self, batch, batch_idx):
        train_data, train_labels = batch
        train_data = train_data
        spk_output, _ = self.model.single_forward(train_data)
        # measure accuracy
        batch_accuracy = self.calc_accuracy(spk_output, train_labels)
        self.log("train_accuracy", batch_accuracy / len(train_labels), prog_bar=True)

        # measure loss
        train_loss = self.loss_fn(spk_output.sum(0), train_labels)
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        validation_data, validation_labels = batch
        validation_data = validation_data

        # Val set forward pass
        spk_output, _ = self.model.single_forward(validation_data)

        # measure accuracy
        batch_accuracy = self.calc_accuracy(spk_output, validation_labels)
        self.log("val_accuracy", batch_accuracy / len(validation_labels), prog_bar=True)

        # measure loss
        val_loss = self.loss_fn(spk_output.sum(0), validation_labels)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_data, test_labels = batch
        test_data = test_data

        test_spk, _ = self.model.single_forward(test_data)

        # measure accuracy
        batch_accuracy = self.calc_accuracy(test_spk, test_labels)
        self.log("test_accuracy", batch_accuracy / len(test_labels), prog_bar=True)

        # measure loss
        test_loss = self.loss_fn(test_spk.sum(0), test_labels)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        test_data, test_labels = batch

        test_spk, _ = self.model.single_forward(test_data)

        _, pred = test_spk.sum(dim=0).max(1)

        return pred.detach().cpu().numpy()

    @staticmethod
    def calc_accuracy(output, labels):
        _, idx = output.sum(dim=0).max(1)
        batch_acc = (labels == idx).sum()  # .detach().cpu().numpy())
        return batch_acc


if __name__ == "__main__":
    batch_size = 256
    lr = 1.0e-3
    dataset_path = "../../../data/data_watch_subset_40.npz"
    seed_everything(42)
    data_module = WISDM(path=dataset_path, batch_size=batch_size)

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps
    #
    spiking_network = SpikingNetwork(lr=lr)
    trainer = Trainer(
        accelerator="auto",
        max_epochs=300,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_checkpointing=True,
        enable_model_summary=True,
    )
    trainer.fit(
        model=spiking_network,
        datamodule=data_module,
        ckpt_path="./model_data/WISDM_snnTorch.ckpt",
    )
    trainer.save_checkpoint("./model_data/WISDM_snnTorch.ckpt")
    test_data = trainer.test(
        model=spiking_network, datamodule=data_module, verbose=False
    )[0]
    print(test_data)
