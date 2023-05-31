"""
=====================================================================
Project:      NeuroBench
File:         benchmark.py
Description:  Python code describing pipeline for benchmarks for the motor prediction task
Date:         12. May 2023
=====================================================================
Copyright stuff
=====================================================================
"""

import torch
import time
import numpy as np
import logging
from enum import Enum
import matplotlib.pyplot as plt
import os
from torch import nn

from neurobench.datasets import Dataset
from neurobench.benchmarks.metrics import compute_r2_score, compute_effective_macs, compute_latency


class TYPE(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class Benchmark:
    """
    Run a Benchmark

    Parameters
    ----------
    dataset :  Dataset
        dataset to benchmark, contains indices of training, validation and testing data
    net     :  nn.Module
        network to benchmark
    hyperparams : dict
        dictionary containing information relevant to the run

    Methods
    ----------
    run
        runs training, validation and testing pipeline
    train
        runs training pipeline
    evaluate
        runs testing / validation pipeline
    """
    def __init__(self, dataset: Dataset, net: nn.Module, hyperparams):
        super().__init__()
        self.dataset = dataset
        self.net = net
        self.hyperparams = hyperparams
        self.result = Result(hyperparams)

        # training parameters
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=hyperparams['lr'],
                                           weight_decay=hyperparams['weight_decay'])
        self.criterion = torch.nn.MSELoss()

    def run(self):
        """
        Iterate over number of epochs and do training, validation and testing of the network

        """
        # iterate over number of epochs
        for epoch in range(self.hyperparams['epochs']):
            # run training loop and update results
            self.result.add_results(epoch, TYPE.TRAINING, *self.train())

            # run validation loop and update results
            self.result.add_results(epoch, TYPE.VALIDATION, *self.evaluate(type=TYPE.VALIDATION))

        # run testing loop and update results
        self.result.add_results(-1, TYPE.TESTING, *self.evaluate(type=TYPE.TESTING))

    def train(self):
        """
        Training pipeline for the benchmark

        """
        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(self.dataset, self.dataset.ind_train),
            batch_size=self.hyperparams['batch_size'],
            shuffle=True)

        # stores (MSE, R2, Effective MAC)
        results = np.zeros(4)
        macs = None

        t0 = time.time()
        for batch, (x, y) in enumerate(train_loader):
            x = x.to(self.hyperparams['device'])
            y = y.to(self.hyperparams['device'])

            out = self.net(x)

            # Spikes are required to compute effective MACs
            prediction, spikes = out

            loss = self.criterion(y, prediction)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update metrics, multiplication with batch_size (i.e. x.shape[0]) in case of smaller batch
            with torch.no_grad():
                current_batch_size = x.shape[0]
                results[0] += loss.item() * current_batch_size
                results[1] += compute_r2_score(y, prediction)
                macs = spikes * current_batch_size if macs is None else macs + spikes * current_batch_size
                results[3] += compute_latency(y, prediction) * current_batch_size

            print("finished Batch {} in {}s".format(batch, time.time()-t0))

        # compute average over number of samples
        results /= len(train_loader.dataset.indices)

        # using the average spiking activity per layer, compute the effective macs
        results[2] = compute_effective_macs(self.net, macs / len(train_loader.dataset.indices))

        return results, time.time() - t0

    def evaluate(self, type: TYPE):
        """
        Validation / Testing pipeline for the benchmark
        Returns
        ----------
        type    : TYPE
            whether training or testing indices should be used
        """
        indices = self.dataset.ind_test if type == TYPE.TESTING else self.dataset.ind_val
        dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(self.dataset, indices),
            batch_size=self.hyperparams['batch_size'],
            shuffle=False)

        # stores (MSE, R2, Effective MAC)
        results = np.zeros(4)
        macs = None

        t0 = time.time()
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.hyperparams['device'])
            y = y.to(self.hyperparams['device'])

            out = self.net(x)

            # Spikes are required to compute effective MACs
            if isinstance(out, tuple):
                prediction, spikes = out
            else:
                prediction, spikes = out, None

            loss = self.criterion(y, prediction)

            # update metrics, multiplication with batch_size (i.e. x.shape[0]) in case of smaller batch
            with torch.no_grad():
                current_batch_size = x.shape[0]
                results[0] += loss.item() * current_batch_size
                results[1] += compute_r2_score(y, prediction)
                macs = spikes * current_batch_size if macs is None else macs + spikes * current_batch_size
                results[3] += compute_latency(y, prediction) * current_batch_size

        # compute average over number of samples
        results /= len(dataloader.dataset.indices)

        # using the average spiking activity per layer, compute the effective macs
        results[2] = compute_effective_macs(self.net, macs / len(dataloader.dataset.indices))

        return results, time.time() - t0


class Result:
    """
    Class containing results and visualizations

    Parameters
    ----------
    hyperparams : dict
        dictionary containing relevant information for the current run

    Methods
    ----------
    add_results
        update results
    visualize_learning_curve
        visualize the mse, r2 and effective macs over epochs

    """
    def __init__(self, hyperparams):
        super().__init__()
        # results for training, validation and testing per epoch
        self.mse = np.zeros((hyperparams['epochs'], 3))
        self.r2 = np.zeros((hyperparams['epochs'], 3))
        self.macs = np.zeros((hyperparams['epochs'], 3))
        self.latency = np.zeros((hyperparams['epochs'], 3))

        # store results in new folder consisting of name and path
        self.path = hyperparams['name'] + "/" + time.strftime("%m-%d-%Y %H:%M:%S", time.localtime())
        os.makedirs("results/" + self.path)

        # create a logger for training
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename="results/" + self.path + "/report.log",
                            filemode='w',
                            level=logging.INFO,
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S')

    def add_results(self, idx, type: TYPE, results, duration):
        """
        Update Logger and Results

        Parameters
        ----------
        idx     :  int
            current epoch
        type    :  TYPE
            enum of training, validation and testing runs
        results : ndarray
            contains mse, r2, latency and effective macs of current epoch
        duration: float
            duration of current epoch
        """
        self.mse[idx, type.value], self.r2[idx, type.value], self.macs[idx, type.value], self.latency[idx, type.value], =\
            results
        self.logger.info("{} Epoch: {} in {}s with Loss L2: {:3.4} R2: {:3.4} MAC {:3.4} Latency: {:3.4}".format(
            type.name, idx, duration, *results))

    def visualize_learning_curve(self, save=True):
        """
        Visualize or Save the learning curves of the various metrics

        Parameters
        ----------
        save     :  bool
            either store or visualize plot
        """
        # plot learning curve over epochs
        plt.subplot(221)
        plt.title("MSE")
        plt.plot(self.mse)

        # plot R2 score over epochs
        plt.subplot(222)
        plt.title("R2")
        plt.plot(self.r2)

        # plot effective MACs per timestep over epochs
        plt.subplot(223)
        plt.title("MAC")
        plt.plot(self.macs)

        # plot effective MACs per timestep over epochs
        plt.subplot(224)
        plt.title("Latency")
        plt.plot(self.latency)

        plt.tight_layout()
        plt.legend(('train', "validation", "test"))

        if save:
            plt.savefig("results/" + self.path + "/results.png")
        else:
            plt.show()
