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

    def run(self, run_num, results_summary):
        
        print("seed: {}".format(self.hyperparams['seed'][run_num]))

        early_stop = utils.EarlyStopping(patience=10, verbose=False)

        best_score = float("-inf")

        for epoch in tqdm(range(self.hyperparams['epochs']), desc="Training Epoch"):
            self.result.add_results(epoch, TYPE.TRAINING, *self.train())
            self.result.add_results(epoch, TYPE.VALIDATION, *self.evaluate(type=TYPE.VALIDATION))

            if self.result.r2[epoch, TYPE.VALIDATION.value].item() > best_score:
                torch.save(self.net.state_dict(), "model_state_dict.pth")
                best_score = self.result.r2[epoch, TYPE.VALIDATION.value].item()
                print("The new best score is {}".format(best_score))

            early_stop(self.result.mse[epoch, TYPE.TRAINING.value], self.net)
            self.lr_scheduler.step()

            if early_stop.early_stop or epoch == self.hyperparams['epochs'] - 1:
                if epoch > 40:
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

        train_pre_total, train_label = None, None

        for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):  # This is the main loop without shuffling
            if idx not in self.dataset.ind_train:
                continue

            if sample.shape[1] >= 8000:
                continue

            train_set = CustomDataset(sample, label)
            train_loader = DataLoader(
                dataset=train_set,
                batch_size=self.hyperparams['batch_size'],
                drop_last=False,
                shuffle=False)
            train_batch = iter(train_loader)

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

                data = torch.permute(data, (0, 2, 1))
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

                h_t = h_curr.detach()
                c_t = c_curr.detach()

                train_count += data.shape[0]

        # R2 calculation
        results[1] = self.result.r2_results(train_pre_total, train_label, model_type=self.model_type,
                                            num_step=self.hyperparams['num_steps'])

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

        test_pre_total, test_label = None, None

        if type == TYPE.TESTING:
            self.net.load_state_dict(torch.load("model_state_dict.pth"))

        with torch.no_grad():
            for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
                if idx not in indices:
                    continue

                if sample.shape[1] >= 8000:
                    continue

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

                    h_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])
                    c_t = torch.randn(1, data.shape[0], self.net.hidden_size).to(self.hyperparams['device'])

                    data = torch.permute(data, (0, 2, 1))
                    pre, (h_eval_curr, c_eval_curr) = self.net(data, h_t, c_t)
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

                    h_t = h_eval_curr
                    c_t = c_eval_curr

                    test_count += data.shape[0]

        # R2 calculation
        results[1] = self.result.r2_results(test_pre_total, test_label, model_type=self.model_type,
                                            num_step=self.hyperparams['num_steps'])

        results[0] = loss_test / test_count
        if type.value == 1:
            print(' Validation loss: {}, R2_score: {}'.format(results[0], results[1]))
        elif type.value == 2:
            print(' Test loss: {}, R2_score: {}'.format(results[0], results[1]))

        return results

    def plot_gt_vs_prediction(self, type):
        folder_path = "./Figures/" + str(type) + "/"

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        self.net.eval()
        self.net.to(self.hyperparams['device'])

        if type == TYPE.TRAINING:
            indices = self.dataset.ind_train
        elif type == TYPE.VALIDATION:
            indices = self.dataset.ind_val
        elif type == TYPE.TESTING:
            indices = self.dataset.ind_test

        with torch.no_grad():
            for idx, (sample, label) in enumerate(zip(self.dataset.samples, self.dataset.labels)):
                if idx not in indices:
                    continue
                if sample.shape[1] >= 8000:
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

                    data = torch.permute(data, (0, 2, 1))
                    pre, (h_curr, c_curr) = self.net(data, h_t, c_t)

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

    def run_results(self, run_num, epoch, final_results):

        final_results[run_num, 0], final_results[run_num, 1] = \
            self.mse[epoch, TYPE.TRAINING.value], self.r2[epoch, TYPE.TRAINING.value]

        final_results[run_num, 2], final_results[run_num, 3] = \
            self.mse[epoch, TYPE.VALIDATION.value], self.r2[epoch, TYPE.VALIDATION.value]

        final_results[run_num, 4], final_results[run_num, 5] = \
            self.mse[-1, TYPE.TESTING.value], self.r2[-1, TYPE.TESTING.value]

        print(final_results)

        return  final_results

