# EEG Motor Imagery Classification with Spiking Neural Networks

An EEG motor imagery classification pipeline using a Spiking Neural Network (SNN), evaluated with NeuroBench.

---

## Dataset — Lee2019\_MI

The dataset is sourced from [MOABB](https://moabb.neurotechx.com/) (`Lee2019_MI`) and consists of EEG recordings from **54 subjects** performing a **2-class motor imagery task** (left hand vs. right hand).

| Property | Value |
| --- | --- |
| Subjects | 54 |
| Sessions per subject | 2 (recorded on different days) |
| Trials per subject per session | 100 (50 per class) |
| Total trials | 10,800 |
| EEG channels | 62 |
| Original sampling rate | 1000 Hz |
| Resampled to | 100 Hz |
| Epoch window | 1.0 s – 3.5 s after trial onset |
| Frequency band | 8 – 30 Hz (mu and beta rhythms) |
| Classes | left hand, right hand |

The two sessions were recorded on separate days, making the dataset suitable for studying both within-session and cross-session generalization.

The preprocessed train/validation splits are automatically downloaded from [HuggingFace](https://huggingface.co/datasets/NeuroBench/thor_eeg_mi) via the `ThorEEGMI` dataset class — no manual preparation or splitting is required.

---

## Folder Structure

```
.
├── Makefile
├── train.py               # Train the EEG-SNN model
└── benchmark.py           # NeuroBench evaluation
```

---

## Pipeline

### 1. `train.py` — Train the SNN

Trains a 3-layer fully connected Spiking Neural Network using [snnTorch](https://snntorch.readthedocs.io/). The model processes EEG timesteps sequentially, one timestep at a time, accumulating spikes across the temporal dimension. The best model by validation accuracy is saved.

The dataset is automatically downloaded to `data` folder.

```
make train
```

| Split | Approx. trials |
| --- | --- |
| Train | ~7,560 |
| Val | ~1,620 |

> **Note:** raw data is saved without normalization. The splits are stratified by subject×session (70 / 15 / 15) with a fixed random seed for full reproducibility.

---

### 2. `benchmark.py` — NeuroBench Evaluation

Evaluates the trained model on the validation set using [NeuroBench](https://neurobench.ai/), reporting both static and workload metrics relevant to neuromorphic hardware deployment.

```
make benchmark
```

---

## Running the Full Pipeline

```
make          # runs train -> benchmark
```

Or step by step:

```
make train
make benchmark
```

Other commands:

```
make help     # show available targets
make clean    # delete dataset folder
```

---

## Requirements

```
pip install moabb snntorch neurobench torch scikit-learn scipy matplotlib pandas numpy
```

---

## Design Decisions

**Dataset split strategy.** A trial-level split stratified by subject x session was chosen over a cross-subject split. While a cross-subject split (where entire subjects are held out for test) is the more rigorous approach for claiming real-world BCI generalisation in a research setting, a stratified trial-level split with a fixed seed is simpler, transparent, and reproducible, making it easier to evaluate and compare submissions consistently.

**Both sessions included.** Each subject's trials from both recording sessions are kept together and assigned to the same split. This exposes the model to within-subject session variability during training, which is realistic since EEG signals shift between recording days.

---

## THOR × NeuroBench Challenge 2026

This repository serves as a baseline for the [THOR × NeuroBench Challenge 2026](https://neuromorphiccommons.com/events/thor_neurob_comp_2026/challenge.html) (March – July 2026), which targets real-world neuromorphic deployment of BCI systems across two tracks:

* **Track 1 — Classification Accuracy:** correctly classifying left vs. right hand motor imagery
* **Track 2 — Compute Efficiency:** minimising synaptic operations (SynOps) and memory footprint for deployment on the THOR neuromorphic architecture

### Additional Preprocessing
 
The data provided via `ThorEEGMI` is already bandpass filtered, resampled, and epoched. If your submission applies any further preprocessing (e.g. normalisation, spatial filtering, feature extraction), it must be declared as a `preprocessors` list passed to the NeuroBench `Benchmark` class — not applied offline to the raw arrays.
 
To implement a custom preprocessor, subclass `NeuroBenchPreProcessor` and implement `__call__`. It receives a `(data, targets)` tuple of PyTorch tensors and must return a `(data, targets)` tuple:
 
```python
from neurobench.processors.preprocessors import NeuroBenchPreProcessor
 
class MyPreprocessor(NeuroBenchPreProcessor):
    def __call__(self, dataset):
        data, targets = dataset
        # apply your transformation to data and/or targets
        return data, targets
```
 
Then pass it to the `Benchmark` class:
 
```python
from neurobench.benchmarks import Benchmark
 
benchmark = Benchmark(
    model=nb_model,
    dataloader=loader,
    preprocessors=[MyPreprocessor()],   # declare all preprocessing here
    postprocessors=[postprocess],
    metric_list=[...],
)
```
 
This ensures all preprocessing steps are tracked, auditable, and applied consistently at evaluation time.

### Reproducibility

All participants are strongly encouraged to seed every source of randomness in their submission. Unreproducible results cannot be fairly compared or verified by the challenge organisers. At minimum, seed the following:

```python
import random, os
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

Call `seed_everything()` at the very top of every script, before any data loading or model initialisation. Report the seed value used in your submission.

### Exporting Dependencies

Participants must include a full list of their dependencies so that results can be reproduced in a clean environment. Use one of the following:

**Option A — `requirements.txt` (pip):**

```
pip freeze > requirements.txt
```

**Option B — `environment.yml` (conda):**

```
conda env export > environment.yml
```

Include the generated file at the root of your submission repository. Submissions without a dependency file may be disqualified if the evaluation environment cannot be reproduced.