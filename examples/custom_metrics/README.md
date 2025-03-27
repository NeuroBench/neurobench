# Creating Custom Metrics in NeuroBench: A Deep Dive

This workshop guide explains how to create custom metrics in the NeuroBench framework, using the AverageActivity metric as a practical example. We'll explore how to implement metrics that can analyze neural network behavior, particularly focusing on spiking neural networks.

## Understanding Metric Types in NeuroBench

NeuroBench supports three types of metrics:

1. **Static Metrics**: Evaluate fixed properties of the model (e.g., parameter count)
2. **Workload Metrics**: Evaluate performance during inference
3. **Accumulated Metrics**: A special type of workload metric that accumulates statistics across batches

In this workshop, we'll focus on creating an AccumulatedMetric, which is particularly useful for analyzing spike related informationas it allows us to gather statistics across multiple batches of data and compute desired metrics at the end.

## The AverageActivity Metric: A Case Study

Let's examine how to create a metric that analyzes the activity patterns of neurons across different layers in a spiking neural network.

We are interested in the average activity of neurons across different layers. We want to compute the min, max, mean, median, and q1 and q3 of the activity of neurons across different layers.

### 1. Setting Up the Metric Class

```python
class AverageActivityMetric(AccumulatedMetric):
    def __init__(self):
        super().__init__(requires_hooks=True)  # Enable activation hooks
        self.layer_activities = defaultdict(list)
        self.num_batches = 0
        self.all_spikes = {}
```

Key points:
- Inherit from `AccumulatedMetric` for batch-wise accumulation
- Set `requires_hooks=True` to access activation information
- Initialize dictionaries to store statistics across batches

### 2. Understanding Hooks

Hooks are essential for accessing spike information in spiking neural networks. They allow us to intercept and record activation values during the forward pass:

```python
def _get_layer_activities(self, model):
    spike_lsts = {}
    for i, hook in enumerate(model.activation_hooks):
        if not hook.activation_outputs:
            continue
        spike_lsts[f'activation_layer_{i+1}'] = hook.activation_outputs
```

The hooks system:
- Automatically captures activation outputs from each layer
- Provides access to spike information during inference
- Allows us to analyze temporal dynamics

### 3. The `__call__` Method: Batch Processing

The `__call__` method is invoked for each batch during benchmarking:

```python
def __call__(self, model, preds, data):
    # Get activities for all layers
    self._get_layer_activities(model)
    self.num_batches += 1
    return self.compute()
```

This method:
- Processes each batch of data
- Accumulates statistics across batches
- Can return intermediate results if needed

### 4. The `compute` Method: Final Calculations

The `compute` method performs the final calculations after all batches are processed:

```python
def compute(self):
    results = {}
    for layer_name, spikes in self.all_spikes.items():
        # Concatenate all batches
        spikes = torch.cat(spikes, dim=1)
        # Calculate statistics
        spike_percentages = ((spikes.sum(dim=0)/spikes.shape[0])
                           .sum(dim=0)/spikes.shape[1])
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
```

This method:
- Aggregates data from all batches
- Computes final statistics
- Returns a structured result dictionary

### 5. Visualization Support

The metric includes a visualization method to help understand the results:

```python
@classmethod
def plot_activity_distributions(cls, results):
    plt.figure(figsize=(12, 6))
    results = results['AverageActivityMetric']
    layer_names = list(results.keys())
    activity_data = [results[layer]['histogram'] for layer in layer_names]
    
    bp = plt.boxplot(activity_data, 
                    labels=layer_names,
                    vert=True,
                    widths=0.5,
                    showmeans=True,
                    meanline=True,
                    patch_artist=True)
```

## Using the Metric in a Benchmark

Here's how to integrate the metric into a NeuroBench benchmark:

```python
from neurobench.benchmarks import Benchmark
from neurobench.metrics.workload import AverageActivityMetric

# Create benchmark with metrics
benchmark = Benchmark(
    model=model,
    dataloader=test_set_loader,
    preprocessors=preprocessors,
    postprocessors=postprocessors,
    metrics=[[], [AverageActivityMetric]]  # Empty list for static metrics
)

# Run benchmark
results = benchmark.run(device=device)

# Visualize results
AverageActivityMetric.plot_activity_distributions(results)
```

## Key Takeaways

1. **Accumulated Metrics**:
   - Perfect for analyzing spiking neural networks
   - Allow gathering statistics across multiple batches
   - Require hooks for accessing activation information

2. **Hook System**:
   - Provides access to spike information
   - Enables analysis of temporal dynamics
   - Works with various spiking neural network architectures

3. **Metric Structure**:
   - `__init__`: Set up hooks and storage
   - `__call__`: Process each batch
   - `compute`: Calculate final statistics
   - `reset`: Clear accumulated data

4. **Best Practices**:
   - Use hooks for accessing network internals
   - Accumulate statistics across batches
   - Provide visualization support
   - Document the metric's purpose and usage

## Further Exploration

Try creating your own custom metrics by:
1. Identifying what aspects of your network you want to analyze
2. Choosing between Static, Workload, or Accumulated metrics
3. Implementing the necessary methods
4. Adding visualization support
5. Integrating with the NeuroBench framework

Remember to:
- Document your metric's purpose
- Handle edge cases
- Provide clear visualization
- Test with different network architectures 