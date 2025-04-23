import torch
import numpy as np
import matplotlib.pyplot as plt

from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from neurobench.metrics.utils.decorators.check_shape import check_shapes
from neurobench.metrics.utils.layers.binary_copy import make_binary_copy
from neurobench.metrics.utils.layers.macs import single_layer_MACs

class RewardScore(AccumulatedMetric):
    '''
    Accumulate rewards over interactions.
    '''
    def __init__(self):
        self.rewards = []

    def __call__(self, model, pred, data):
        self.rewards.append(data[1])

    def compute(self, plot=True):
        avg = np.mean(self.rewards)
        std = np.std(self.rewards)

        sorted_rewards = sorted(self.rewards)
        num_lowest = int(len(sorted_rewards) * 0.05)
        lowest_rewards = sorted_rewards[:num_lowest]
        risk = np.mean(lowest_rewards)

        if plot:
            # Plot histogram of rewards
            plt.hist(self.rewards, bins=75)
            plt.title('Histogram of Rewards')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.show()

        return {'avg': avg, 'std': std, 'risk': risk}

class AverageTime(AccumulatedMetric):
    '''
    Accumulate times over interactions.
    '''
    def __init__(self):
        self.times = []

    def __call__(self, model, pred, data):
        self.times.append(data[0])

    def compute(self):
        return np.mean(self.times)