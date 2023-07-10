# NeuroBench API: Developent Version
This is a brief overview of the state of the NeuroBench API, the specifications NeuroBench components are expected to conform to, and any exceptions or errata that have come up during development.  
  
Example Python snippets may be shown here and will be kept up-to-date.


## Overview
| **Component** | **Status** | **Known Errata** |
|---------------|------------|------------------|
| Data          | Frozen     |                  |
| Dataset       | Frozen     |                  |
| Processor     | Frozen     |                  |
| Model         | Tentative  |                  |
| Metrics       | Proposed   |                  |
| Benchmark     | Tentative  |                  |

### **Status**:
- **Frozen**: Work can be started with the expectation that the I/O around that component is defined and will not change.
- **Tentative**: Almost frozen, work can be started but functions may need to be added, renamed, etc.
- **Proposed**: Implement this at your own risk.

<div class="page"/>

## Specifications
### **Data:**
```
Format:
    tensor: A PyTorch tensor of shape (batch, timesteps, ...)
```
### **Dataset:**
```
Output:
    (data, targets): A tuple of PyTorch tensors. The first dimension (batch) is expected to match.
```
### **Processor:**
```
Input:
    (data, targets): A tuple of PyTorch tensors. The first dimension (batch) is expected to match.
Output:
    (data, targets): A tuple of PyTorch tensors. The first dimension (batch) is expected to match.
```
```python
class Processor:
    def __init__(self):
		...
    def __call__(self, dataset):
		...

alg = Processor()
new_dataset = alg(dataset) # dataset: (data, targets)
```

<div class="page"/>

### **Model\*:**
```
Input:
    data: A PyTorch tensor of shape (batch, timesteps, ...)
Output:
    preds: A PyTorch tensor of shape [TODO].
```
```python
class NeurobenchModel:
    def __init__(self, net):
		...
    def __call__(self, batch):
		...

model = SNNTorchModel(net)
pred = model(batch)
output = (pred, targets) # Follow dataset format
```
### **Metrics\*:**
```
TODO
```

<div class="page"/>

### **Benchmark\*:**
```
Input:
    model: The model to be tested. Must be wrapped in a NeuroBench class.
    test_set: The dataset of form (data, targets) to be inferred.
    pre_processors: A list of processor functions.
    post_processors: A list of post-processor functions.
    metrics: A list of strings. The names of the metric will be used to call it from the metrics file. User defined metrics should be discouraged.
Output:
    results: Either a dict or specific NeuroBenchResults class.
```
```python
class Benchmark:
    def __init__(self, model, test_set, pre_processors, post_processors, metrics):
		...
    def run(self):
		...

benchmark = Benchmark(
	model, 
	test_set,
	[processor1, processor2],
    [processor3, processor4], 
	["accuracy", "model_size"]
)

```

**\* =** Tentative or Proposed API

<div class="page"/>

## Known Errata
Any anomalies that break the high-level API will be noted here but attempts will be made to keep this to a minimum.