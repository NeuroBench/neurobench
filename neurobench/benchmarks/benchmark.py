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

from neurobench import utils
from neurobench.datasets.primate_reaching import PrimateReaching

class TYPE(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2

class Benchmark():
    def __init__(self, dataset: PrimateReaching, net: nn.Module, hyperparams, model_type="ANN", train_ratio=0.8, delay=0):
        super().__init__()
        self.dataset = dataset
        self.net = net
        self.hyperparams = hyperparams
        self.result = Result(hyperparams)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=hyperparams['lr'],
                                           betas=(0.9, 0.999), weight_decay=hyperparams['weight_decay'])
        self.criterion = torch.nn.MSELoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=64)
        self.model_type = model_type
        self.train_ratio = train_ratio
        self.delay = delay

        # self.multiply_ops, self.add_ops = utils.operation_calculation(self.net, self.model_type)

        # self.sparsity = []

        if self.hyperparams['d_type'] == "torch.float":
            self.d_type = torch.float
        elif self.hyperparams['d_type'] == "torch.double":
            self.d_type = torch.double

    def run(self):
        # for run_num in range(self.hyperparams['max_run']):
        for run_num in range(1):
            torch.manual_seed(self.hyperparams['seed'][run_num])
            print("seed: {}".format(self.hyperparams['seed'][run_num]))

            early_stop = utils.EarlyStopping(patience=7, verbose=False)

            for epoch in tqdm(range(self.hyperparams['epochs']), desc="Training Epoch"):
                self.result.add_results(epoch, TYPE.TRAINING, *self.train())
                self.result.add_results(epoch, TYPE.VALIDATION, *self.evaluate(type=TYPE.VALIDATION))

                early_stop(self.result.r2[epoch, TYPE.VALIDATION.value], self.net)
                self.lr_scheduler.step()

                if early_stop.early_stop or epoch == self.hyperparams['epochs'] - 1:
                    final_epoch = epoch
                    break
            self.result.add_results(-1, TYPE.TESTING, *self.evaluate(type= TYPE.TESTING))

            self.result.final_mean_results(run_num, final_epoch, self.hyperparams)

        if self.hyperparams['save_data']:
            utils.save_results(self.result.re_final_mean, self.train_ratio, self.delay,
                               self.hyperparams['filename'], self.hyperparams['data_save_path'])

        # self.ops_and_sparsity()

    def train(self):
        self.net.train()
        self.net.to(self.hyperparams['device'])
        trainloader = DataLoader(
            dataset=Subset(self.dataset, self.dataset.ind_train),
            batch_size=self.hyperparams['batch_size'],
            drop_last=True,
            shuffle=True)
        train_batch = iter(trainloader)

        results = torch.zeros(2)
        loss_train = 0
        train_count = 0

        for i, (data, target) in enumerate(train_batch):
            data = data.to(self.hyperparams['device'])
            target = target.to(self.hyperparams['device'])
            # print("Input Data Dimension is: {}".format(data.size()))
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
                spk_train, mem_train = self.net(data)

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


        results[0] = loss_train / train_count
        print('Training loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results

    def evaluate(self, type):
        self.net.eval()
        self.net.to(self.hyperparams['device'])
        indices = self.dataset.ind_test if type == TYPE.TESTING else self.dataset.ind_val
        testloader = DataLoader(
            dataset=Subset(self.dataset, indices),
            batch_size=self.hyperparams['batch_size'],
            drop_last=True,
            shuffle=True)
        test_batch = iter(testloader)

        results = torch.zeros(2)
        loss_test = 0
        test_count = 0
        r2score_test = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_batch):
                data = data.to(self.hyperparams['device'])
                target = target.to(self.hyperparams['device'])

                if self.model_type == "ANN":
                    pre = self.net(data.view(self.hyperparams['batch_size'], -1))

                    loss_val = self.criterion(pre[0], target)
                    loss_test += loss_val.item()

                    if i == 0:
                        test_pre_total = pre[0]
                        test_label = target
                    else:
                        test_pre_total = torch.cat((test_pre_total, pre[0]), dim=0)
                        test_label = torch.cat((test_label, target), dim=0)


                elif self.model_type == "SNN":
                    net_results = self.net(data)

                    loss_val_step = torch.zeros((1), dtype=self.d_type, device=self.hyperparams['device'])
                    skl_r2_step = torch.zeros((1), dtype=self.d_type, device=self.hyperparams['device'])
                    for step in range(self.hyperparams['num_steps']):
                        loss_val_step += self.criterion(net_results[-1][step], target)
                        skl_r2_step += r2_score(target.detach().cpu().numpy(), net_results[-1][step].detach().cpu().numpy())
                    r2score_test += (skl_r2_step / self.hyperparams['num_steps'])
                    loss_test += (loss_val_step / self.hyperparams['num_steps'])

                    if i == 0:
                        test_pre_total = net_results[-1]
                        test_label = target
                    else:
                        test_pre_total = torch.cat((test_pre_total, net_results[-1]), dim=1)
                        test_label = torch.cat((test_label, target), dim=0)

                test_count += 1

        # R2 calculation
        if self.model_type == "ANN":
            results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type)
        elif self.model_type == "SNN":
            results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type,
                                                num_step=self.hyperparams['num_steps'])

        results[0] = loss_test / test_count
        if type.value == 1:
            print(' Validation loss: {}, R2_score: {}'.format(results[0], results[1]))
        elif type.value == 2:
            print(' Test loss: {}, R2_score: {}'.format(results[0], results[1]))
        # print('V_skl_r2 : {}'.format(r2score_test/test_count))

        return results
    
    def ops_and_sparsity(self):
        self.net.eval()
        indices = self.dataset.ind_test
        testloader = DataLoader(
            dataset=Subset(self.dataset, indices),
            batch_size=self.hyperparams['batch_size'],
            drop_last=True,
            shuffle=True)
        test_batch = iter(testloader)
        sparsity = [[], [], [], []] # order is final_layer, input, first_layer, second_layer
        with torch.no_grad():
            # data, target = next(test_batch)
            for data, target in test_batch:
                data = data.to(self.hyperparams['device'])
                target = target.to(self.hyperparams['device'])
                eval_output = self.net(data)
                if self.model_type == "ANN":
                    sparsity_result = utils.sparsity_calculation("ANN", eval_output)
                elif self.model_type == "SNN":
                    sparsity_result = utils.sparsity_calculation("SNN", eval_output)
                for i in range(len(sparsity)):
                    sparsity[i].append(sparsity_result[i])

        average_sparsity = []
        for i in range(len(sparsity)):
            current_layer_ave = sum(sparsity[i])/len(sparsity[i])
            average_sparsity.append(current_layer_ave)
            if i == 0:
                print("Final Layer's Average Output Sparsity is: {}".format(current_layer_ave))
            elif i == 1:
                print("Input Average Sparsity is: {}".format(current_layer_ave))
            else:
                print("Layer-{}'s Average Sparsity is: {}".format(i-1, current_layer_ave))

        ops = [[], []] # Ops contains (multiplies, adds)
        test_batch = iter(testloader)
        with torch.no_grad():
            # data, target = next(test_batch)
            # for data, target in test_batch:
            #     data = data.to(self.hyperparams['device'])
            #     target = target.to(self.hyperparams['device'])
            #     ops_result = utils.operation_calculation(self.net, "ANN", average_sparsity)
            #     # ops_result = utils.operation_calculation(self.net, "SNN", average_sparsity)
            #     for i in range(len(ops)):
            #         ops[i].append(ops_result[i])
            data, target = next(test_batch)
            data = data.to(self.hyperparams['device'])
            target = target.to(self.hyperparams['device'])
            if self.model_type == "ANN":
                ops_result = utils.operation_calculation(self.net, "ANN", sparsity=average_sparsity)
            elif self.model_type == "SNN":
                # outputs = self.net(data.view(self.hyperparams['batch_size'], -1))
                ops_result = utils.operation_calculation(self.net, "SNN", sparsity=average_sparsity)
            for i in range(len(ops)):
                ops[i].append(ops_result[i])
        
        multiplies_ave = sum(ops[0])/len(ops[0])
        add_ave = sum(ops[1])/len(ops[1])
        print("Average No. of Multiply Operations: {}".format(multiplies_ave))
        print("Average No. of Addition Operations: {}".format(add_ave))


class Result():
    def __init__(self, hyperparams):
        self.mse = np.zeros((hyperparams['epochs'], 3))
        self.r2 = np.zeros((hyperparams['epochs'], 3))
        self.final_results = torch.zeros((hyperparams['max_run'], 6), device=hyperparams['device'])
        self.re_final_mean = None

    @staticmethod
    def r2_results(pre_data, target_data, model_type="ANN", num_step=None):

        if model_type == "ANN":
            X_numerator = torch.sum((target_data[:, 0] - pre_data[:, 0])**2)
            Y_numerator = torch.sum((target_data[:, 1] - pre_data[:, 1])**2)
            X_original_label_mean = torch.mean(target_data[:, 0])
            Y_original_label_mean = torch.mean(target_data[:, 1])
            X_denominator = torch.sum((target_data[:, 0] - X_original_label_mean)**2)
            Y_denominator = torch.sum((target_data[:, 1] - Y_original_label_mean)**2)
            X_r2 = 1- (X_numerator/ X_denominator)
            Y_r2 = 1- (Y_numerator/ Y_denominator)
            r2 = (X_r2 + Y_r2)/2

        elif model_type == "SNN":
            r2_step = 0
            for step in range(num_step):
                X_numerator = torch.sum((target_data[:, 0] - pre_data[step][:, 0])**2)
                Y_numerator = torch.sum((target_data[:, 1] - pre_data[step][:, 1])**2)
                X_original_label_mean = torch.mean(target_data[:, 0])
                Y_original_label_mean = torch.mean(target_data[:, 1])
                X_denominator = torch.sum((target_data[:, 0] - X_original_label_mean)**2)
                Y_denominator = torch.sum((target_data[:, 1] - Y_original_label_mean)**2)
                X_r2 = 1- (X_numerator/ X_denominator)
                Y_r2 = 1- (Y_numerator/ Y_denominator)
                r2_step += (X_r2 + Y_r2)/2

            r2 = r2_step / num_step

        return r2

    def add_results(self, epoch, type: TYPE, *results):
        self.mse[epoch, type.value], self.r2[epoch, type.value] = results

    def final_mean_results(self, run_num, epoch, hyperparams):

        self.final_results[run_num, 0], self.final_results[run_num, 1] = \
            self.mse[epoch, TYPE.TRAINING.value], self.r2[epoch, TYPE.TRAINING.value]

        self.final_results[run_num, 2], self.final_results[run_num, 3] = \
            self.mse[epoch, TYPE.VALIDATION.value], self.r2[epoch, TYPE.VALIDATION.value]

        self.final_results[run_num, 4], self.final_results[run_num, 5] = \
            self.mse[-1, TYPE.TESTING.value], self.r2[-1, TYPE.TESTING.value]

        print(self.final_results)
        self.re_final_mean = torch.mean(self.final_results, dim=0)
        print(self.re_final_mean)

