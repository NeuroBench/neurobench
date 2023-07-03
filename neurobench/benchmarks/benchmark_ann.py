"""
"""
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import time
import numpy as np
import os
from torch import nn
from tqdm import tqdm
from enum import Enum
from sklearn.metrics import r2_score
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import math
import random

from neurobench import utils
from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.datasets.dataset import Dataset, CustomDataset

import matplotlib.pyplot as plt

class TYPE(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2

class BenchmarkANN():
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
        self.min_len_segment = None

        if self.hyperparams['d_type'] == "torch.float":
            self.d_type = torch.float
        elif self.hyperparams['d_type'] == "torch.double":
            self.d_type = torch.double

    def run(self, run_num, results_summary):
        
        print("seed: {}".format(self.hyperparams['seed'][run_num]))

        early_stop = utils.EarlyStopping(patience=7, verbose=False)

        for epoch in tqdm(range(self.hyperparams['epochs']), desc="Training Epoch"):
            self.result.add_results(epoch, TYPE.TRAINING, *self.train())
            self.result.add_results(epoch, TYPE.VALIDATION, *self.evaluate(type=TYPE.VALIDATION))
            
            early_stop(self.result.mse[epoch, TYPE.VALIDATION.value], self.net)
            self.lr_scheduler.step()

            if early_stop.early_stop or epoch == self.hyperparams['epochs'] - 1:
                if epoch > 10:
                    final_epoch = epoch
                    break
        self.result.add_results(-1, TYPE.TESTING, *self.evaluate(type=TYPE.TESTING))

        # Plot the predicted velocity against the ground truth
        # self.plot_gt_vs_prediction(TYPE.TRAINING)
        # self.plot_gt_vs_prediction(TYPE.VALIDATION)
        # self.plot_gt_vs_prediction(TYPE.TESTING)

        # Calculate the no. of ops & sparsity
        # self.ops_and_sparsity()

        return self.result.run_results(run_num, final_epoch, results_summary)

    def train(self):
        self.net.train()
        self.net.to(self.hyperparams['device'])

        results = torch.zeros(2)
        loss_train = 0
        train_count = 0
        train_set_sample = []
        train_set_label = []

        train_pre_total, train_label = None, None

        shuffle_index = list(range(0, len(self.dataset.samples)))
        self.min_len_segment = self.dataset.samples[0].shape[1]

        for idx in shuffle_index:

            if idx not in self.dataset.ind_train:
                continue

            sample = self.dataset.samples[idx]
            label = self.dataset.labels[idx]
            
            train_set_sample.append(sample)
            train_set_label.append(label)

            if sample.shape[1] < self.min_len_segment:
                self.min_len_segment = sample.shape[1]

        train_samp = torch.cat(train_set_sample, dim=1)
        train_lab = torch.cat(train_set_label, dim=1)
        train_set = CustomDataset(train_samp, train_lab)
        train_loader = DataLoader(
                dataset=train_set,
                batch_size=self.hyperparams['batch_size'],
                drop_last=False,
                shuffle=True)
        train_batch = iter(train_loader)

        for i, (data, target) in enumerate(train_batch):
            if data.shape[0] <= 15:
                continue

            data = data.to(self.hyperparams['device'])
            target = target.to(self.hyperparams['device'])

            if self.model_type == "ANN":
                pre = self.net(data)

                loss_val = self.criterion(pre, target)
                loss_train += loss_val.item()

                if i==0 and train_count == 0:
                    train_pre_total = pre
                    train_label = target
                else:
                    train_pre_total = torch.cat((train_pre_total, pre), dim=0)
                    train_label = torch.cat((train_label, target), dim=0)
            elif self.model_type == "SNN":
                spk_train, mem_train, pre = self.net(data)

                loss_val = self.criterion(pre, target)

                loss_train += loss_val.item()

                if i==0 and train_count == 0:
                    train_pre_total = pre
                    train_label = target
                else:
                    train_pre_total = torch.cat((train_pre_total, pre), dim=0)
                    train_label = torch.cat((train_label, target), dim=0)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            train_count += data.shape[0]

        # R2 calculation
        if self.model_type == "ANN":
            results[1] = self.result.r2_results(train_pre_total, train_label, model_type=self.model_type)
        elif self.model_type == "SNN":
            # results[1] = self.result.r2_results(train_pre_total, train_label, model_type=self.model_type,
            #                                     num_step=self.hyperparams['num_steps'])
            results[1] = self.result.r2_results(train_pre_total, train_label, model_type="ANN")

        results[0] = loss_train / train_count
        print(' Training loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results

    def evaluate(self, type):
        self.net.eval()
        self.net.to(self.hyperparams['device'])
        indices = self.dataset.ind_test if type == TYPE.TESTING else self.dataset.ind_val

        results = torch.zeros(2)
        loss_test = 0
        test_count = 0
        r2score_test = 0
        test_set_sample = []
        test_set_label = []

        test_pre_total, test_label = None, None
        shuffle_index = torch.randperm(len(self.dataset.samples))

        with torch.no_grad():
            
            for idx in shuffle_index:

                if idx not in indices:
                    continue

                sample = self.dataset.samples[idx]
                label = self.dataset.labels[idx]

                if sample.shape[1] < self.min_len_segment:
                    self.min_len_segment = sample.shape[1]

                test_set = CustomDataset(sample, label)
                test_loader = DataLoader(
                    dataset=test_set,
                    batch_size=self.hyperparams['batch_size'],
                    drop_last=False,
                    shuffle=False)
                test_batch = iter(test_loader)

                
                for i, (data, target) in enumerate(test_batch):
                    data = data.to(self.hyperparams['device'])
                    target = target.to(self.hyperparams['device'])

                    if data.shape[0] <= 15:
                        pad_size = 16 - data.shape[0]
                        data_padding_order = (0,0,0,0,0,pad_size)
                        data = torch.nn.functional.pad(data, data_padding_order, "constant", 0)
                        target_padding_order = (0,0,0,pad_size)
                        target = torch.nn.functional.pad(target, target_padding_order, "constant", 0)

                    if self.model_type == "ANN":
                        pre = self.net(data)

                        loss_val = self.criterion(pre[0], target)
                        loss_test += loss_val.item()

                        if i == 0 and test_count == 0:
                            test_pre_total = pre[0]
                            test_label = target
                        else:
                            test_pre_total = torch.cat((test_pre_total, pre[0]), dim=0)
                            test_label = torch.cat((test_label, target), dim=0)
                    elif self.model_type == "SNN":
                        spk_test, mem_test, pre = self.net(data)

                        loss_val = self.criterion(pre, target)
                        loss_test += loss_val.item()

                        if i==0 and test_count == 0:
                            test_pre_total = pre
                            test_label = target
                        else:
                            test_pre_total = torch.cat((test_pre_total, pre), dim=0)
                            test_label = torch.cat((test_label, target), dim=0)

                    test_count += data.shape[0]

        # Fliter
        test_pre_total_f = utils.bessel_lowpass_filter(data=test_pre_total.t().detach().cpu().numpy(), cutoff=0.05, order=4)
        test_pre_total_f = test_pre_total_f.copy()
        test_pre_total_f = torch.tensor(test_pre_total_f).t().to(self.hyperparams['device'])
        test_pre_total = test_pre_total_f

        # R2 calculation
        if self.model_type == "ANN":
            results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type)
        elif self.model_type == "SNN":
            # results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type,
            #                                     num_step=self.hyperparams['num_steps'])
            results[1] = self.result.r2_results(test_pre_total, test_label, model_type="ANN")

        results[0] = loss_test / test_count
        if type.value == 1:
            print(' Validation loss: {}, R2_score: {}'.format(results[0], results[1]))
        elif type.value == 2:
            print(' Test loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results

    def plot_gt_vs_prediction(self, type):
        self.net.eval()
        self.net.to(self.hyperparams['device'])

        if type == TYPE.TRAINING:
            indices = self.dataset.ind_train
        elif type == TYPE.VALIDATION:
            indices = self.dataset.ind_val
        elif type == TYPE.TESTING:
            indices = self.dataset.ind_test

        test_set_sample = []
        test_set_label = []
        log_x_sum = []
        log_y_sum = []
        log_y_filter_sum = []
        log_y_gt_sum = []
        
        with torch.no_grad():
            for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
                if idx not in indices:
                    continue

                curr_set = CustomDataset(sample, label)
                loader = DataLoader(
                    dataset=curr_set,
                    batch_size=self.hyperparams['batch_size'],
                    drop_last=False,
                    shuffle=False)
                loader_batch = iter(loader)

                x_vel_pred, y_vel_pred = [], []
                x_vel_gt, y_vel_gt = [], []

                for i, (data, target) in enumerate(loader_batch):
                    if i >= 100:
                        break

                    data = data.to(self.hyperparams['device'])
                    target = target.to(self.hyperparams['device'])

                    if data.shape[0] <= 15:
                        pad_size = 16 - data.shape[0]
                        data_padding_order = (0,0,0,0,0,pad_size)
                        data = torch.nn.functional.pad(data, data_padding_order, "constant", 0)
                        target_padding_order = (0,0,0,pad_size)
                        target = torch.nn.functional.pad(target, target_padding_order, "constant", 0)

                    if self.model_type == "ANN":
                        pre = self.net(data)
                    elif self.model_type == "SNN":
                        spk_train, mem_train, pre = self.net(data)

                    x_vel_pred.append(pre[0][:, 0])
                    y_vel_pred.append(pre[0][:, 1])
                    x_vel_gt.append(target[:, 0])
                    y_vel_gt.append(target[:, 1])

                x_vel_pred_plot = torch.concat(x_vel_pred).detach().cpu().numpy()
                y_vel_pred_plot = torch.concat(y_vel_pred).detach().cpu().numpy()
                x_vel_gt_plot = torch.concat(x_vel_gt).detach().cpu().numpy()
                y_vel_gt_plot = torch.concat(y_vel_gt).detach().cpu().numpy()

                y_vel_pred_plot_filter = utils.bessel_lowpass_filter(data=y_vel_pred_plot, cutoff=0.05, order=4)
                y_vel_pred_plot_filter = y_vel_pred_plot_filter.copy()

                fft_y = np.abs(fft(y_vel_pred_plot[: self.min_len_segment]))
                fft_y_filter = np.abs(fft(y_vel_pred_plot_filter[: self.min_len_segment]))
                fft_y_gt = np.abs(fft(y_vel_gt_plot[: self.min_len_segment]))
                N = self.min_len_segment
                xf = fftfreq(N)
                log_y = [np.log(r) for r in fft_y]
                log_y_filter = [math.log(r) for r in fft_y_filter]
                log_y_gt = [math.log(r) for r in fft_y_gt]
                log_y_sum.append(log_y)
                log_y_filter_sum.append(log_y_filter)
                log_y_gt_sum.append(log_y_gt)
            
            average_log_y = np.mean(np.array(log_y_sum), axis=0)
            average_log_y_filter = np.mean(np.array(log_y_filter_sum), axis=0)
            average_log_y_gt = np.mean(np.array(log_y_gt_sum), axis=0)
            fig = plt.figure()
            plt.plot(xf, average_log_y, color="r", label="Predicted Y Velocity Fourier Transform")
            plt.plot(xf, average_log_y_filter, color="g", label="Predicted Y Velocity (with filter) Fourier Transform")
            plt.plot(xf, average_log_y_gt, color="b", label="Y GT Velocity Fourier Transform")
            plt.legend(loc="best")
            plt.xlabel("Data Point Frequency")
            plt.ylabel("Logarithm")
            plt.savefig("./Figures/{}_{}_{}_Vel.jpg".format(idx, i, type))
            plt.close()

    def ops_and_sparsity(self):
        self.net.eval()
        indices = self.dataset.ind_test
        sparsity = [[], [], [], []] # order is final_layer, input, first_layer, second_layer
        with torch.no_grad():
            for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
                if idx not in indices:
                    continue
                curr_set = CustomDataset(sample, label)
                loader = DataLoader(
                    dataset=curr_set,
                    batch_size=self.hyperparams['batch_size'],
                    drop_last=False,
                    shuffle=False)
                loader_batch = iter(loader)
                for data, target in loader_batch:
                    data = data.to(self.hyperparams['device'])
                    target = target.to(self.hyperparams['device'])
                    eval_output = self.net(data)
                    if self.model_type == "ANN":
                        sparsity_result = sparsity_calculation("ANN", eval_output)
                    elif self.model_type == "SNN":
                        sparsity_result = sparsity_calculation("SNN", eval_output)
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
        
        with torch.no_grad():

            curr_set = CustomDataset(self.dataset.samples[0], self.dataset.labels[0])
            loader = DataLoader(
                dataset=curr_set,
                batch_size=self.hyperparams['batch_size'],
                drop_last=False,
                shuffle=False)
            loader_batch = iter(loader)
            for data, target in loader_batch:
                data = data.to(self.hyperparams['device'])
                target = target.to(self.hyperparams['device'])
                if self.model_type == "ANN":
                    ops_result = operation_calculation(self.net, "ANN", sparsity=average_sparsity)
                elif self.model_type == "SNN":
                    ops_result = operation_calculation(self.net, "SNN", sparsity=average_sparsity)
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

        elif model_type == "LSTM" or model_type == "GRU" or model_type == "RNN":
            X_numerator = torch.sum((target_data[:, 0] - pre_data[:, 0])**2)
            Y_numerator = torch.sum((target_data[:, 1] - pre_data[:, 1])**2)
            X_original_label_mean = torch.mean(target_data[:, 0])
            Y_original_label_mean = torch.mean(target_data[:, 1])
            X_denominator = torch.sum((target_data[:, 0] - X_original_label_mean)**2)
            Y_denominator = torch.sum((target_data[:, 1] - Y_original_label_mean)**2)
            X_r2 = 1- (X_numerator/ X_denominator)
            Y_r2 = 1- (Y_numerator/ Y_denominator)
            r2 = (X_r2 + Y_r2)/2

        return r2

    def add_results(self, epoch, type: TYPE, *results):
        self.mse[epoch, type.value], self.r2[epoch, type.value] = results

    def run_results(self, run_num, epoch, final_results):

        final_results[run_num, 0], final_results[run_num, 1] = \
            self.mse[epoch, TYPE.TRAINING.value], self.r2[epoch, TYPE.TRAINING.value]

        final_results[run_num, 2], final_results[run_num, 3] = \
            self.mse[epoch, TYPE.VALIDATION.value], self.r2[epoch, TYPE.VALIDATION.value]

        final_results[run_num, 4], final_results[run_num, 5] = \
            self.mse[-1, TYPE.TESTING.value], self.r2[-1, TYPE.TESTING.value]

        print(final_results)

        return  final_results
    
def operation_calculation(net, model_type: str, sparsity: list=[], spikes=[]):
    # Sparsity List contains the following:
    # [layer3's sparsity, input sparsity, layer1's sparsity, layer2's sparsity]

    multiplies, adds = 0, 0
    if model_type == "ANN":
        for param in net.named_parameters():
            print(param[0], param[1].size()[::-1])
            local_dim = param[1].size()[::-1]
            if "weight" in param[0]:
                temp = 1
                for dim in local_dim:
                    print("DIM: {}".format(dim))
                    temp *= dim
                
                if "batchnorm" not in param[0]:
                    if "1" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[1]))
                        temp *= (sparsity[1])
                    elif "2" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[2]))
                        temp *= (sparsity[2])
                    elif "3" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[3]))
                        temp *= (sparsity[3])
                
                multiplies += temp
                if "batchnorm" not in param[0]:
                    adds += temp
            elif "bias"in param[0]:
                temp = 1
                for dim in local_dim:
                    temp *= dim

                adds += temp
            print("Current Mult: {}.\nCurrent Add: {}".format(multiplies, adds))
            
    elif model_type == "SNN":
        ns = net.num_step
        for param in net.named_parameters():
            print(param[0], param[1].size()[::-1])
            local_dim = param[1].size()[::-1]
            if "weight" in param[0]:
                temp = 1
                for dim in local_dim:
                    print("DIM: {}".format(dim))
                    temp *= dim
                
                if "batchnorm" not in param[0]:
                    if "1" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[1])) # Sparsiy here implies (No of spikes/timestep) zero elements on average
                        temp *= (1-sparsity[1])*ns
                    elif "2" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[2]))
                        temp *= (1-sparsity[2])*ns
                    elif "3" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[3]))
                        temp *= (1-sparsity[3])*ns
                    adds += temp    # Addition Operations for the Weight's being applied to the Spiking Neuron
                else:
                    multiplies += temp*ns   # This is for Batch Norm's division
                    
            elif "bias" in param[0]:
                temp = 1
                for dim in local_dim:
                    temp *= dim
                temp *= ns
                adds += temp
                if "batchnorm" not in param[0]:
                    multiplies += temp
                    adds += temp

            print("Current Mult: {}.\nCurrent Add: {}".format(multiplies, adds))
    else:
        print("Invalid Input Model Type. Can only be ANN or SNN, got {} instead!".format(model_type))

    print("Total Number of Multiply Ops is: {}".format(multiplies))
    print("Total Number of Add Ops is: {}".format(adds))

    return multiplies, adds

def sparsity_calculation(model_type: str, model_outputs: tuple, num_step=1):
    layers_sparsity = [0 for _ in range(len(model_outputs))]
    
    if model_type == "ANN":
        for i in range(len(model_outputs)):
            sparsity = 1 - (len(torch.nonzero(model_outputs[i])) / torch.numel(model_outputs[i]))
            if i == 0:
                print("Final output sparsity is: {}".format(sparsity))
            elif i == 1:
                print("Input Sparsity is: {}".format(sparsity))
            else:
                print("Layer-{}'s Sparsity is: {}".format(i-1, sparsity))
            layers_sparsity[i] = sparsity

    elif model_type == "SNN":
        for i in range(len(model_outputs)):
            if i == 1:
                sparsity = 1 - (len(torch.nonzero(model_outputs[i])) / torch.numel(model_outputs[i]))
                print("Input Sparsity is: {}".format(sparsity))
            elif i == len(model_outputs)-1:
                continue
            else:
                local_sparsity = []
                for step in range(num_step):
                    sparsity = 1 - (len(torch.nonzero(model_outputs[i][step])) / torch.numel(model_outputs[i][step]))
                    local_sparsity.append(sparsity)
                sparsity = sum(local_sparsity)/len(local_sparsity)
                if i == 0:
                    print("Final output sparsity is: {}".format(sparsity))
                else:
                    print("Layer-{}'s Sparsity is: {}".format(i-1, sparsity))
            layers_sparsity[i] = sparsity
    else:
        print("Invalid Model Type. Should be either ANN or SNN, got {}".format(model_type))
    return layers_sparsity

def input_operations_calculation(data):
    print("Input Operation Calculation:")
    print(data.size())
    print(torch.mean(data))

