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

from neurobench import utils
from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.datasets.dataset import Dataset, CustomDataset

import matplotlib.pyplot as plt

class TYPE(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2

class BenchmarkLSTM():
    def __init__(self, dataset: PrimateReaching, net: nn.Module, hyperparams, model_type="ANN", train_ratio=0.8, delay=0):
        super().__init__()
        self.dataset = dataset
        self.net = net
        self.hyperparams = hyperparams
        self.result = Result(hyperparams)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=hyperparams['lr'],
                                           betas=(0.9, 0.999), weight_decay=hyperparams['weight_decay'])
        self.criterion = torch.nn.MSELoss()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=32)
        self.model_type = model_type
        self.train_ratio = train_ratio
        self.delay = delay

        if self.hyperparams['d_type'] == "torch.float":
            self.d_type = torch.float
        elif self.hyperparams['d_type'] == "torch.double":
            self.d_type = torch.double

    def run(self, run_num, results_summary, k_fold=False, fold_num=0):
        
        # print("seed: {}".format(self.hyperparams['seed'][run_num]))

        early_stop = utils.EarlyStopping(patience=10, verbose=False)

        best_score = float("-inf")

        for epoch in tqdm(range(self.hyperparams['epochs']), desc="Training Epoch"):
        # for epoch in range(self.hyperparams['epochs']):
            self.result.add_results(epoch, TYPE.TRAINING, *self.train(k_fold, fold_num))
            self.result.add_results(epoch, TYPE.VALIDATION, *self.evaluate(type=TYPE.VALIDATION, k_fold=k_fold, fold_num=fold_num))

            if self.result.r2[epoch, TYPE.VALIDATION.value].item() > best_score:
                torch.save(self.net.state_dict(), "model_state_dict.pth")
                best_score = self.result.r2[epoch, TYPE.VALIDATION.value].item()
                # print("The new best score is {}".format(best_score))

            early_stop(self.result.mse[epoch, TYPE.TRAINING.value], self.net)
            self.lr_scheduler.step()

            if early_stop.early_stop or epoch == self.hyperparams['epochs'] - 1:
                final_epoch = epoch
                if epoch > 40:
                    break
        self.result.add_results(-1, TYPE.TESTING, *self.evaluate(type=TYPE.TESTING, k_fold=k_fold, fold_num=fold_num))

        # Plot the predicted velocity against the ground truth
        # self.plot_gt_vs_prediction(TYPE.TRAINING, k_fold, fold_num)
        # self.plot_gt_vs_prediction(TYPE.VALIDATION, k_fold, fold_num)
        self.plot_gt_vs_prediction(TYPE.TESTING, k_fold, fold_num)

        # Calculate the no. of ops & sparsity
        self.ops_and_sparsity(k_fold, fold_num)

        return self.result.run_results(run_num, final_epoch, results_summary, self.hyperparams, fold_num)

    def train(self, k_fold=False, fold_num=0):
        self.net.train()
        self.net.to(self.hyperparams['device'])

        results = torch.zeros(2)
        loss_train = 0
        train_count = 0

        train_pre_total, train_label = None, None

        ind_train = self.dataset.ind_train if not k_fold else self.dataset.ind_train[fold_num]

        # for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
        for idx in ind_train:
            self.dataset.segment_no = idx
            # train_batch = self.batch_generator(sample, label)
        
            train_batch = DataLoader(dataset=self.dataset,
                                     batch_size=self.hyperparams['batch_size'],
                                     drop_last=False,
                                     shuffle=False)

            for i, (data, target) in enumerate(train_batch):
                data = data.to(self.hyperparams['device'])
                target = target.to(self.hyperparams['device'])

                if data.shape[0] <= 15:
                    pad_size = 16 - data.shape[0]
                    # The padding order goes:
                    # (Last dim's start, last dim's end, 2nd last dim's start, 2nd last dim's end, ..., first dim's start, first dim's end)
                    data_padding_order = (0,0,0,0,0,pad_size)
                    data = torch.nn.functional.pad(data, data_padding_order, "constant", 0)
                    target_padding_order = (0,0,0,pad_size)
                    target = torch.nn.functional.pad(target, target_padding_order, "constant", 0)

                h_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])
                c_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])

                # data = torch.permute(data, (0, 2, 1))
                pre, (h_curr, c_curr) = self.net(data, h_t, c_t)
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

                # h_t = h_curr.detach()
                # c_t = c_curr.detach()

            train_count += 1

        # R2 calculation
        results[1] = self.result.r2_results(train_pre_total, train_label, model_type=self.model_type,
                                            num_step=self.hyperparams['num_steps'])

        results[0] = loss_train / train_count
        # print(' Training loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results

    def evaluate(self, type, k_fold=False, fold_num=0):
        self.net.eval()
        self.net.to(self.hyperparams['device'])

        dataset_type = "test" if type == TYPE.TESTING else "validation"
        if k_fold:
            indices = self.dataset.ind_test[fold_num] if type == TYPE.TESTING else self.dataset.ind_val[fold_num]
        else:
            indices = self.dataset.ind_test if type == TYPE.TESTING else self.dataset.ind_val

        results = torch.zeros(2)
        loss_test = 0
        test_count = 0
        r2score_test = 0

        test_pre_total, test_label = None, None

        if type == TYPE.TESTING:
            self.net.load_state_dict(torch.load("model_state_dict.pth"))

        with torch.no_grad():
            # for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
            for idx in indices:
                self.dataset.segment_no = idx

                # test_batch = self.batch_generator(sample, label)
                test_batch = DataLoader(dataset=self.dataset,
                                        batch_size=self.hyperparams['batch_size'],
                                        drop_last=False,
                                        shuffle=False)

                for i, (data, target) in enumerate(test_batch):
                    data = data.to(self.hyperparams['device'])
                    target = target.to(self.hyperparams['device'])

                    if data.shape[0] <= 15:
                        pad_size = 16 - data.shape[0]
                        data_padding_order = (0,0,0,0,0,pad_size)
                        data = torch.nn.functional.pad(data, data_padding_order, "constant", 0)
                        target_padding_order = (0,0,0,pad_size)
                        target = torch.nn.functional.pad(target, target_padding_order, "constant", 0)

                    h_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])
                    c_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])

                    # data = torch.permute(data, (0, 2, 1))
                    pre, (h_eval_curr, c_eval_curr) = self.net(data, h_t, c_t)[:2]
                    h_eval_curr.detach()
                    c_eval_curr.detach()
                    
                    loss_val = self.criterion(pre, target)
                    loss_test += loss_val.item()

                    if i==0 and test_count == 0:
                        test_pre_total = pre
                        test_label = target
                    else:
                        test_pre_total = torch.cat((test_pre_total, pre), dim=0)
                        test_label = torch.cat((test_label, target), dim=0)

                    # h_t = h_eval_curr
                    # c_t = c_eval_curr

                    test_count += data.shape[0]

        # R2 calculation
        results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type,
                                            num_step=self.hyperparams['num_steps'])

        results[0] = loss_test / test_count
        # if type.value == 1:
        #     print(' Validation loss: {}, R2_score: {}'.format(results[0], results[1]))
        # elif type.value == 2:
        #     print(' Test loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results

    def plot_gt_vs_prediction(self, type, k_fold: bool=False, fold_num: int=0):
        folder_path = "./Figures/" + str(type) + "/"

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        self.net.eval()
        self.net.to(self.hyperparams['device'])

        dataset_type = "train" if type == TYPE.TRAINING else ("test" if type == TYPE.TESTING else "validation")

        if type == TYPE.TRAINING:
            indices = self.dataset.ind_train if not k_fold else self.dataset.ind_train[fold_num]
        elif type == TYPE.VALIDATION:
            indices = self.dataset.ind_val if not k_fold else self.dataset.ind_val[fold_num]
        elif type == TYPE.TESTING:
            indices = self.dataset.ind_test if not k_fold else self.dataset.ind_test[fold_num]

        with torch.no_grad():
            # for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
            for idx in indices:
                self.dataset.segment_no = idx

                # loader_batch = self.batch_generator(sample, label)
                loader_batch = DataLoader(dataset=self.dataset,
                                          batch_size=self.hyperparams['batch_size'],
                                          drop_last=False,
                                          shuffle=False)

                x_vel_pred, y_vel_pred = [], []
                x_vel_gt, y_vel_gt = [], []

                for i, (data, target) in enumerate(loader_batch):
                    if i >= 100:
                        break

                    data = data.to(self.hyperparams['device'])
                    target = target.to(self.hyperparams['device'])

                    # Pad data & target with zeros when the batch size is
                    # less than the padlen of the Bessel Filter, which is 15
                    if data.shape[0] <= 15:
                        pad_size = 16 - data.shape[0]
                        data_padding_order = (0,0,0,0,0,pad_size)
                        data = torch.nn.functional.pad(data, data_padding_order, "constant", 0)
                        target_padding_order = (0,0,0,pad_size)
                        target = torch.nn.functional.pad(target, target_padding_order, "constant", 0)

                    h_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])
                    c_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])

                    # data = torch.permute(data, (0, 2, 1))
                    pre, (h_curr, c_curr), *_ = self.net(data, h_t, c_t)

                    h_t = h_curr.detach()
                    c_t = c_curr.detach()

                    x_vel_pred.append(pre[:, 0])
                    y_vel_pred.append(pre[:, 1])
                    x_vel_gt.append(target[:, 0])
                    y_vel_gt.append(target[:, 1])

                x_vel_pred_plot = torch.concat(x_vel_pred).detach().cpu().numpy()
                y_vel_pred_plot = torch.concat(y_vel_pred).detach().cpu().numpy()
                x_vel_gt_plot = torch.concat(x_vel_gt).detach().cpu().numpy()
                y_vel_gt_plot = torch.concat(y_vel_gt).detach().cpu().numpy()

                fig = plt.figure()
                plt.plot(x_vel_pred_plot, color="r", label="Predicted X Velocity")
                plt.plot(x_vel_gt_plot, color="b", label="Ground Truth X Velocity")
                plt.legend(loc="best")
                plt.xlabel("Data Point")
                plt.ylabel("Velocity cm/s")
                plt.savefig(folder_path + "{}_{}_X_Vel.jpg".format(idx, i))
                plt.close()

                fig = plt.figure()
                plt.plot(y_vel_pred_plot, color="r", label="Predicted Y Velocity")
                plt.plot(y_vel_gt_plot, color="b", label="Ground Truth Y Velocity")
                plt.legend(loc="best")
                plt.xlabel("Data Point")
                plt.ylabel("Velocity cm/s")
                plt.savefig(folder_path + "{}_{}_Y_Vel.jpg".format(idx, i))
                plt.close()

    def batch_generator(self, sample, label):
        current_set = CustomDataset(sample, label)
        current_loader = DataLoader(
            dataset=current_set,
            batch_size=self.hyperparams['batch_size'],
            drop_last=False,
            shuffle=False)
        current_batch = iter(current_loader)
        return current_batch

    def ops_and_sparsity(self, k_fold:bool=False, fold_num:int=0):
        self.net.eval()
        if k_fold:
            indices = self.dataset.ind_test[fold_num]
        else:
            indices = self.dataset.ind_test
        # order is final_layer, input, first_layer, second_layer
        # sparsity = [[], [], [], []]
        sparsity = []
        with torch.no_grad():
            # for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
            for idx in range(len(self.dataset.samples)):
                self.segment_no = idx
                loader_batch = DataLoader(dataset=self.dataset,
                                          batch_size=self.hyperparams['batch_size'],
                                          drop_last=False,
                                          shuffle=False)
                for data, target in loader_batch:
                    data = data.to(self.hyperparams['device'])
                    target = target.to(self.hyperparams['device'])

                    if data.shape[0] <= 15:
                        pad_size = 16 - data.shape[0]
                        data_padding_order = (0,0,0,0,0,pad_size)
                        data = torch.nn.functional.pad(data, data_padding_order, "constant", 0)
                        target_padding_order = (0,0,0,pad_size)
                        target = torch.nn.functional.pad(target, target_padding_order, "constant", 0)

                    h_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])
                    c_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])
                    eval_output = self.net(data, h_t, c_t)
                    sparsity_result = self.sparsity_calculation(eval_output)
                    if len(sparsity) == 0:
                        for i in range(len(sparsity_result)):
                            sparsity.append([])
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

        # Need to account for sparsity in my ops calculation now
        ops_result = self.operation_calculation(sparsity=average_sparsity)

        with open(self.hyperparams["results_save_path"] + self.hyperparams['filename'] + "_ops_sparsity_result.csv", "a") as f:
            f.write("Multiplies, {}, Adds, {}\n".format(ops_result[0], ops_result[1]))
            sparsity_string = ""
            for i, sparse in enumerate(average_sparsity):
                if i == 0:
                    sparsity_string += "Final Layer's Average Sparsity, {},".format(sparse)
                elif i == 1:
                    sparsity_string += "Input Layer's Average Sparsity, {},".format(sparse)
                else:
                    sparsity_string += "Layer-{}'s Average Sparsity, {},".format(i-1, sparse)
            f.write(sparsity_string + "\n")

        print("Average No. of Multiply Operations: {}".format(ops_result[0]))
        print("Average No. of Addition Operations: {}".format(ops_result[1]))

    def sparsity_calculation(self, model_outputs: tuple, num_step: int=1):
        if self.model_type == "LSTM":
            layers_sparsity = [0 for _ in range(len(model_outputs)-1)]
        else:
            layers_sparsity = [0 for _ in range(len(model_outputs))]
    
        if self.model_type == "ANN":
            for i in range(len(model_outputs)):
                sparsity = 1 - (len(torch.nonzero(model_outputs[i])) / torch.numel(model_outputs[i]))
                # if i == 0:
                    # print("Final output sparsity is: {}".format(sparsity))
                # elif i == 1:
                    # print("Input Sparsity is: {}".format(sparsity))
                # else:
                    # print("Layer-{}'s Sparsity is: {}".format(i-1, sparsity))
                layers_sparsity[i] = sparsity
        elif self.model_type == "SNN":
            for i in range(len(model_outputs)):
                if i == 1:
                    sparsity = 1 - (len(torch.nonzero(model_outputs[i])) / torch.numel(model_outputs[i]))
                    # print("Input Sparsity is: {}".format(sparsity))
                elif i == len(model_outputs)-1:
                    continue
                else:
                    local_sparsity = []
                    for step in range(num_step):
                        sparsity = 1 - (len(torch.nonzero(model_outputs[i][step])) / torch.numel(model_outputs[i][step]))
                        local_sparsity.append(sparsity)
                    sparsity = sum(local_sparsity)/len(local_sparsity)
                    # if i == 0:
                        # print("Final output sparsity is: {}".format(sparsity))
                    # else:
                        # print("Layer-{}'s Sparsity is: {}".format(i-1, sparsity))
                layers_sparsity[i] = sparsity
        elif self.model_type == "LSTM":
            for i in range(len(model_outputs)):
                if i == 1:
                    # We skip the hidden states h & c
                    continue
                sparsity = 1 - (len(torch.nonzero(model_outputs[i])) / torch.numel(model_outputs[i]))
                if i == 0:
                    # print("Final output sparsity is: {}".format(sparsity))
                    layers_sparsity[i] = sparsity
                elif i == 2:
                    # print("Input Sparsity is: {}".format(sparsity))
                    layers_sparsity[i-1] = sparsity
                else:
                    # print("Layer-{}'s Sparsity is: {}".format(i-2, sparsity))
                    layers_sparsity[i-1] = sparsity
        else:
            print("Invalid Model Type. Should be either ANN or SNN, got {}".format(self.model_type))
        return layers_sparsity

    def operation_calculation(self, sparsity=[]):
        multiplies, adds = 0, 0
        if self.model_type == "ANN":
            for param in self.net.named_parameters():
                # print(param[0], param[1].size()[::-1])
                local_dim = param[1].size()[::-1]
                if "weight" in param[0]:
                    temp = 1
                    for dim in local_dim:
                        temp *= dim

                    if "batchnorm" not in param[0]:
                        if "1" in param[0]:
                            # print("Local Sparsity is: {}".format(sparsity[1]))
                            temp *= (1 - sparsity[1])
                        elif "2" in param[0]:
                            # print("Local Sparsity is: {}".format(sparsity[2]))
                            temp *= (1 - sparsity[2])
                        elif "3" in param[0]:
                            # print("Local Sparsity is: {}".format(sparsity[3]))
                            temp *= (1 - sparsity[3])

                    multiplies += temp
                    if "batchnorm" not in param[0]:
                        adds += temp
                elif "bias"in param[0]:
                    temp = 1
                    for dim in local_dim:
                        temp *= dim
                    adds += temp
        elif self.model_type == "SNN":
            print("SNN Ops")
        elif self.model_type == "LSTM":
            for param in self.net.named_parameters():
                # print(param[0], param[1].size()[::-1])

                local_dim = param[1].size()[::-1]
                if "weight" in param[0]:
                    if "hh" in param[0]:
                        continue
                    temp = 1
                    # if "conv" in param[0]:
                    #     stride = net.conv.stride[0]
                    if "lstm" in param[0]:
                        in_dim, out_dim = local_dim[0], int(local_dim[1]/4)

                        multiplies += (1-sparsity[3])*(4*(2*in_dim*out_dim + out_dim) + 4*out_dim)
                        adds += (1-sparsity[3])*(4*(2*in_dim*out_dim + 2*out_dim) + out_dim)
                    else:
                        for dim in local_dim:
                            temp *= dim

                        if "batchnorm" not in param[0]:
                            if "norm_layer" in param[0]:
                                temp *= (1-sparsity[1])
                            elif "conv" in param[0]:
                                temp *= (1-sparsity[2])
                            elif "fc" in param[0]:
                                temp *= (1-sparsity[-1])

                        multiplies += temp

                        if "batchnorm" not in param[0]:
                            adds += temp
                elif "bias"in param[0]:
                    if "hh" in param[0]:
                        continue
                    temp = 1
                    if "lstm" in param[0]:
                        continue
                    else:
                        for dim in local_dim:
                            temp *= dim
                        adds += temp

        print("Total Number of Multiply Ops is: {}".format(multiplies))
        print("Total Number of Add Ops is: {}".format(adds))

        return multiplies, adds

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

        elif model_type == "LSTM":
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

    def run_results(self, run_num, epoch, final_results, hyperparams, fold_num=0):
        final_results[run_num*hyperparams["fold_num"]+fold_num, 0], final_results[run_num*hyperparams["fold_num"]+fold_num, 1] = \
            self.mse[epoch, TYPE.TRAINING.value], self.r2[epoch, TYPE.TRAINING.value]

        final_results[run_num*hyperparams["fold_num"]+fold_num, 2], final_results[run_num*hyperparams["fold_num"]+fold_num, 3] = \
            self.mse[epoch, TYPE.VALIDATION.value], self.r2[epoch, TYPE.VALIDATION.value]

        final_results[run_num*hyperparams["fold_num"]+fold_num, 4], final_results[run_num*hyperparams["fold_num"]+fold_num, 5] = \
            self.mse[-1, TYPE.TESTING.value], self.r2[-1, TYPE.TESTING.value]

        print(final_results)

        return  final_results

