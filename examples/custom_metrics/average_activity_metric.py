import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

from neurobench.metrics.abstract.workload_metric import AccumulatedMetric


class AverageActivityMetric(AccumulatedMetric):
    """
    A workload metric that computes activity statistics for each layer in a neural network.
    For each layer, it tracks the min, max, mean, median, and q1 and q3 of the activity of its neurons.
    
    This metric is particularly useful for analyzing:
    1. Layer-wise activation patterns
    2. Neuron activity distribution
    3. Potential dead neurons or saturation
    4. Layer-specific sparsity patterns
    """
    
    def __init__(self):
        super().__init__(requires_hooks=True)
        self.layer_activities = defaultdict(list)  # Store activities for each layer
        self.num_batches = 0
        self.all_spikes = {}
        self.min_activities = {}
        self.max_activities = {}
        self.mean_activities = {}
        self.q1_activities = {}
        self.q3_activities = {}
        
    def _get_layer_activities(self, model) -> Dict[str, torch.Tensor]:
        """
        Extract activities from all layers in the model.
        This method should be customized based on the model architecture.
        
        Args:
            model: The neural network model
            
        Returns:
            activity of each neuron in each layer as a dictionary
        """
        spike_lsts = {}
        for i, hook in enumerate(model.activation_hooks):
            if not hook.activation_outputs:
                continue

            spike_lsts[f'activation_layer_{i+1}'] = hook.activation_outputs

        activities = {}
        for layer_name, spikes in spike_lsts.items():
            if layer_name not in self.all_spikes:
                self.all_spikes[layer_name] = [torch.stack(spikes, dim=0)]
            else:
                self.all_spikes[layer_name].append(torch.stack(spikes, dim=0))
            # activities[layer_name] = torch.stack(spikes, dim=0).sum(dim=0)/len(spikes) # sum over entire sequence and divide by sequence length
        return activities, len(spikes)
    
    def __call__(self, model, preds, data):
        """
        Accumulate activity statistics for each layer.
        
        Args:
            model: The neural network model
            preds: Model predictions (not used in this metric)
            data: Input data (not used in this metric)
        """
        # Get activities for all layers
        self._get_layer_activities(model)

        # calculate the min, max, mean, q1, q3 of the activities

        
        # Process each layer's activities
        
        self.num_batches += 1

        return self.compute()
    
    def compute(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute the final activity statistics for each layer.
        
        Returns:
            Dictionary containing statistics for each layer:
            {
                'layer_name': {
                    'mean': mean activity per neuron,
                    'std': standard deviation of activities,
                    'histogram': histogram of activities
                }
            }
        """
        results = {}
        
        for layer_name, spikes in self.all_spikes.items():
            # cat all batches
            spikes = torch.cat(spikes, dim=1)
            spike_percentages = ((spikes.sum(dim=0)/spikes.shape[0]).sum(dim=0)/spikes.shape[1]).cpu().detach().numpy() # first sum over batch, then over sequence length
            results[layer_name] = {
                'min': spike_percentages.min(),
                'max': spike_percentages.max(),
                'mean': spike_percentages.mean(),
                'std': spike_percentages.std(),
                'median': np.median(spike_percentages),
                'q1': np.percentile(spike_percentages, 25),
                'q3': np.percentile(spike_percentages, 75),
            }
        return results
    
    def reset(self):
        """Reset the accumulated statistics."""
        self.layer_activities.clear()
        self.num_batches = 0
        # self.all_spikes = {}
    
    def plot_activity_distributions(self, results: Dict[str, Dict[str, np.ndarray]]):
        """
        Plot activity distributions for each layer.
        
        Args:
            results: Results from compute() method
        """
        num_layers = len(results)
        fig, axes = plt.subplots(num_layers, 1, figsize=(10, 4*num_layers))
        
        if num_layers == 1:
            axes = [axes]
        
        for (layer_name, stats), ax in zip(results.items(), axes):
            hist, bins = stats['histogram']
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            ax.bar(bin_centers, hist, width=np.diff(bins)[0])
            ax.set_title(f'Activity Distribution - {layer_name}')
            ax.set_xlabel('Average Activity')
            ax.set_ylabel('Count')
            
            # Add mean and std information
            mean = np.mean(stats['mean'])
            std = np.mean(stats['std'])
            ax.text(0.02, 0.98, f'Mean: {mean:.3f}\nStd: {std:.3f}',
                   transform=ax.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()