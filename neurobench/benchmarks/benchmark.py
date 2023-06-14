"""
"""
import torch
from torch.utils.data import DataLoader, Subset
import time
import numpy as np
import os
from torch import nn
from tqdm import tqdm
from enum import Enum
from sklearn.metrics import r2_score

from neurobench.datasets import Dataset
from neurobench.datasets.primate_reaching import PrimateReaching


class TYPE(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2

class Benchmark():
    def __init__(self, dataset: PrimateReaching, net: nn.Module, hyperparams, model_type="ANN"):
        super().__init__()
        self.dataset = dataset
        self.net = net
        self.hyperparams = hyperparams
        self.result = Result(hyperparams)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=hyperparams['lr'],
                                           betas=(0.9, 0.999), weight_decay=hyperparams['weight_decay'])
        self.criterion = torch.nn.MSELoss()
        self.model_type = model_type

        if self.hyperparams['d_type'] == "torch.float":
            self.d_type = torch.float
        elif self.hyperparams['d_type'] == "torch.double":
            self.d_type = torch.double

    def run(self):
        for epoch in tqdm(range(self.hyperparams['epochs']), desc="Training Epoch"):
            self.result.add_results(epoch, TYPE.TRAINING, *self.train())
            self.result.add_results(epoch, TYPE.VALIDATION, *self.evaluate(type=TYPE.VALIDATION))
        self.result.add_results(-1, TYPE.TESTING, *self.evaluate(type= TYPE.TESTING))

    def train(self):
        self.net.train()
       
        trainloader = DataLoader(
            dataset=Subset(self.dataset, self.dataset.ind_train),
            batch_size=self.hyperparams['batch_size'],
            drop_last=True,
            shuffle=False)
        train_batch = iter(trainloader)

        results = torch.zeros(2)
        loss_train = 0
        train_count = 0

        for i, (data, target) in enumerate(train_batch):
            data = data.to(self.hyperparams['device'])
            target = target.to(self.hyperparams['device'])

            if self.model_type == "ANN":
                pre = self.net(data.view(self.hyperparams['batch_size'], -1))

                loss_val = self.criterion(pre, target)
                loss_train += loss_val.item()

                if i==0:
                    train_pre_total = pre
                    train_label = target
                else:
                    train_pre_total = torch.cat((train_pre_total, pre), dim=0)
                    train_label = torch.cat((train_label, target), dim=0)

            elif self.model_type == "SNN":
                spk_train, mem_train = self.net(data.view(self.hyperparams['batch_size'], -1))

                loss_val_step = torch.zeros((1), dtype=self.d_type, device=self.hyperparams['device'])
                for step in range(self.hyperparams['num_steps']):
                    loss_val_step += self.criterion(mem_train[step], target)
                loss_val = loss_val_step / self.hyperparams['num_steps']
                loss_train += loss_val.item()

                if i==0:
                    train_pre_total = mem_train
                    train_label = target
                else:
                    train_pre_total = torch.cat((train_pre_total, mem_train), dim=1)
                    train_label = torch.cat((train_label, target), dim=0)

            elif self.model_type == "LSTM":
        
                assert len(data.size()) == 3, f"Data needs to be 3D for model {self.model_type}"

                pre = self.net(data)

                loss_val = self.criterion(pre, target)
                loss_train += loss_val.item()

                if i==0:
                    train_pre_total = pre
                    train_label = target
                else:
                    train_pre_total = torch.cat((train_pre_total, pre), dim=0)
                    train_label = torch.cat((train_label, target), dim=0)

            else:
                raise ValueError(f"Code doesn't support {self.model_type} model yet")

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            train_count += 1

        # R2 calculation
        if self.model_type == "ANN":
            results[1] = self.result.r2_results(train_pre_total, train_label, model_type=self.model_type)
        elif self.model_type == "SNN":
            results[1] = self.result.r2_results(train_pre_total, train_label, model_type=self.model_type,
                                                num_step=self.hyperparams['num_steps'])
        elif self.model_type == "LSTM":
            results[1] = self.result.r2_results(train_pre_total, train_label, model_type=self.model_type,
                                                num_step=self.hyperparams['num_steps'])
        else:
            raise ValueError(f"Code doesn't support {self.model_type} model yet")


        results[0] = loss_train / train_count
        print(' Training loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results

    def evaluate(self, type):
        self.net.eval()

        indices = self.dataset.ind_test if type == TYPE.TESTING else self.dataset.ind_val

        testloader = DataLoader(
            dataset=Subset(self.dataset, indices),
            batch_size=self.hyperparams['batch_size'],
            drop_last=True,
            shuffle=False)
        test_batch = iter(testloader)

        results = torch.zeros(2)
        loss_test = 0
        test_count = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_batch):
                data = data.to(self.hyperparams['device'])
                target = target.to(self.hyperparams['device'])

                if self.model_type == "ANN":
                    pre = self.net(data.view(self.hyperparams['batch_size'], -1))

                    loss_val = self.criterion(pre, target)
                    loss_test += loss_val.item()

                    if i == 0:
                        test_pre_total = pre
                        test_label = target
                    else:
                        test_pre_total = torch.cat((test_pre_total, pre), dim=0)
                        test_label = torch.cat((test_label, target), dim=0)

                elif self.model_type == "SNN":
                    spk_test, mem_test = self.net(data.view(self.hyperparams['batch_size'], -1))

                    loss_val_step = torch.zeros((1), dtype=self.d_type, device=self.hyperparams['device'])
                    for step in range(self.hyperparams['num_steps']):
                        loss_val_step += self.criterion(mem_test[step], target)
                    loss_val = loss_val_step / self.hyperparams['num_steps']
                    loss_test += loss_val.item()

                    if i == 0:
                        test_pre_total = mem_test
                        test_label = target
                    else:
                        test_pre_total = torch.cat((test_pre_total, mem_test), dim=1)
                        test_label = torch.cat((test_label, target), dim=0)


                elif self.model_type == "LSTM":
                    pre = self.net(data)

                    loss_val = self.criterion(pre, target)
                    loss_test += loss_val.item()

                    if i==0:
                        test_pre_total = pre
                        test_label = target
                    else:
                        test_pre_total = torch.cat((test_pre_total, pre), dim=0)
                        test_label = torch.cat((test_label, target), dim=0)

                else:
                    raise ValueError(f"Code doesn't support {self.model_type} model yet")

                test_count += 1

        # R2 calculation
        if self.model_type == "ANN":
            results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type)
        elif self.model_type == "SNN":
            results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type,
                                                num_step=self.hyperparams['num_steps'])
        elif self.model_type == "LSTM":
            results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type,
                                                num_step=self.hyperparams['num_steps'])
        else:
            raise ValueError(f"Code doesn't support {self.model_type} model yet")

        results[0] = loss_test / test_count
        if type == TYPE.TESTING:
            print(' Test loss: {}, R2_score: {}'.format(results[0], results[1]))
        else:
            print(' Validation loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results


class Result():
    def __init__(self, hyperparams):
        self.mse = np.zeros((hyperparams['epochs'], 3))
        self.r2 = np.zeros((hyperparams['epochs'], 3))

    @staticmethod
    def r2_results(pre_data, target_data, model_type="ANN", num_step=None):

        if model_type == "ANN":
            numerator = torch.sum((target_data - pre_data)**2)
            original_label_mean = torch.mean(target_data)
            denominator = torch.sum((target_data - original_label_mean)**2)
            r2 = 1- (numerator/ denominator)

        elif model_type == "SNN":
            r2_step = 0
            for step in range(num_step):
                numerator = torch.sum((target_data - pre_data[step]) ** 2)
                original_label_mean = torch.mean(target_data)
                denominator = torch.sum((target_data - original_label_mean) ** 2)
                r2_step += (1 - (numerator / denominator))

            r2 = r2_step / num_step

        elif model_type == "LSTM":

            numerator = torch.sum((target_data - pre_data)**2)
            original_label_mean = torch.mean(target_data)
            denominator = torch.sum((target_data - original_label_mean)**2)
            
            r2 = 1 - (numerator/ denominator)

        else:
            raise ValueError(f"Code doesn't support {self.model_type} model yet")


        return r2

    def add_results(self, epoch, type: TYPE, *results):
        self.mse[epoch, type.value], self.r2[epoch, type.value] = results
