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
                'histogram': spike_percentages
            }
        return results
    
    def reset(self):
        """Reset the accumulated statistics."""
        self.layer_activities.clear()
        self.num_batches = 0
        # self.all_spikes = {}
    
    @classmethod
    def plot_activity_distributions(cls, results: Dict[str, Dict[str, np.ndarray]]):
        """
        Plot boxplots of activity distributions for each layer side by side on one figure.
        
        Args:
            results: Results from compute() method
        """
        # Create a single figure
        plt.figure(figsize=(12, 6))
        
        results = results['AverageActivityMetric']
        # Prepare data for plotting
        layer_names = list(results.keys())
        activity_data = [results[layer]['histogram'] for layer in layer_names]
        
        # Create boxplot
        bp = plt.boxplot(activity_data, 
                        labels=layer_names,
                        vert=True,
                        widths=0.5,
                        showmeans=True,
                        meanline=True,
                        patch_artist=True)
        
        # Customize the plot
        plt.title('Average Activity of Neurons per Layer')
        plt.xlabel('Layer')
        plt.ylabel('Activity')
        plt.ylim(0, 0.5)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        plt.show()