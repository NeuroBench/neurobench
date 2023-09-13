import snntorch as snn
from snntorch import utils
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from neurobench.datasets.WISDM_data_loader import WISDMDataModule
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_steps):
        super().__init__()
        beta = 0.5

        # Initialize layers

        self.net = nn.Sequential(
            nn.Conv2d(6, 80, 3, padding=3),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, init_hidden=True),
            nn.Conv2d(80, 128, kernel_size=2, padding=1),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, init_hidden=True),
            nn.Flatten(),
            nn.Linear(33, 7),
            snn.Leaky(beta=beta, init_hidden=True, output=True),
        )

        self.num_steps = num_steps

    def forward(self, input):
        spk_out, mem_out = self.net(input.reshape(input.shape[1], input.shape[0], 1))
        return spk_out, mem_out

    def single_forward(self, input):
        # Initialize hidden states and outputs at t=0
        mem_rec = []
        spk_rec = []

        utils.reset(self.net)
        for step in range(self.num_steps):
            spk_out, mem_out = self.net(input[step].reshape(input[step].shape[1], input[step].shape[0], 1))

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class SpikingNetwork(pl.LightningModule):
    def __init__(self, num_steps, lr):
        super().__init__()
        self.model = ConvNet(num_steps=num_steps)
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
        train_data = train_data.swapaxes(1, 0)
        spk_output, _ = self.model(train_data)
        # measure accuracy
        batch_accuracy = self.calc_accuracy(spk_output, train_labels)
        self.log('train_accuracy', batch_accuracy / len(train_labels))

        # measure loss
        train_loss = self.loss_fn(spk_output.sum(0), train_labels)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        validation_data, validation_labels = batch
        validation_data = validation_data.swapaxes(1, 0)

        # Val set forward pass
        spk_output, _ = self.model(validation_data)

        # measure accuracy
        batch_accuracy = self.calc_accuracy(spk_output, validation_labels)
        self.log('val_accuracy', batch_accuracy / len(validation_labels))

        # measure loss
        val_loss = self.loss_fn(spk_output.sum(0), validation_labels)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_data, test_labels = batch
        test_data = test_data.swapaxes(1, 0)

        test_spk, _ = self.model(test_data)

        # measure accuracy
        batch_accuracy = self.calc_accuracy(test_spk, test_labels)
        self.log('test_accuracy', batch_accuracy / len(test_labels))

        # measure loss
        test_loss = self.loss_fn(test_spk.sum(0), test_labels)
        self.log('test_loss', test_loss)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        test_data, test_labels = batch
        test_data = test_data.swapaxes(1, 0)

        test_spk, _ = self.model(test_data)

        _, pred = test_spk.sum(dim=0).max(1)

        return pred.detach().cpu().numpy()

    @staticmethod
    def calc_accuracy(output, labels):
        _, idx = output.sum(dim=0).max(1)
        batch_acc = (labels == idx).sum()  # .detach().cpu().numpy())
        return batch_acc


if __name__ == '__main__':
    batch_size = 128
    lr = 1.e-3
    dataset_path = "download your dataset from https://github.com/neuromorphic-polito/NeHAR/blob/main/data/data_watch_subset2_40.npz and store it into a folder"
    data_module = WISDMDataModule(dataset_path, batch_size=batch_size)
    data_module.setup('test')

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps
    #
    spiking_network = SpikingNetwork(lr=lr, num_steps=num_steps)
    trainer = Trainer(accelerator='gpu', max_epochs=3, num_sanity_val_steps=0, enable_progress_bar=False,
                      enable_checkpointing=True,
                      )
    trainer.fit(model=spiking_network, datamodule=data_module)
    test_data = trainer.test(model=spiking_network, datamodule=data_module, verbose=False)[0]
