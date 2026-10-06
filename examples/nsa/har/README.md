# NSA Human Activity Recognition (HAR) baseline

The Human Activity Recognition task from the
[Neuromorphic Sequential Arena](https://arxiv.org/abs/2505.22035) (NSA), with NSA's
LIF spiking baseline and NeuroBench metrics.

## Task

| Property | Value |
| --- | --- |
| Source | WISDM smartphone and smartwatch dataset, watch gyroscope |
| Input | 3 channels (x, y, z), 200 time steps per window (stride 100) |
| Classes | 18 activities |
| Split | NSA's published 80/20 split (`x_train.npy`, `x_test.npy`, ...) |
| Readout | output averaged over time, then argmax |

The data loader downloads NSA's preprocessed split (four `.npy` files) from
[Hugging Face](https://huggingface.co/datasets/liyc5929/neuroseqbench/tree/main/neuromorphic_sequential_arena/WISDM)
into `data/nsa/har/` at the repo root. Using NSA's own split keeps accuracy comparable
with the NSA paper.

## Model

`model.py` builds the same network as NSA's `FFSNN` with `--neuron lif`: three
Linear to LIF layers followed by a linear classifier, using `Recurrent_LIF` and the
triangle surrogate gradient from `neuroseqbench`. Parameter names match NSA's, so a
checkpoint trained with NSA's own script loads directly with `load_model`.

| Config | Hidden sizes | Decay | lr | Grad clip |
| --- | --- | --- | --- | --- |
| Feedforward (default) | 128, 256, 256 | 0.9 | 3e-3 | 1.0 |
| Recurrent (`--recurrent`) | 128, 176, 176 | 0.1 | 1.5e-3 | off |

Both use threshold 0.5, surrogate width 0.6, AdamW, StepLR (step 10, gamma 0.8), batch
256 and 100 epochs, as in NSA's `HAR/run_all.sh`.

## Running

Install NeuroBench and the NSA package:

```
pip install neurobench
pip install git+https://github.com/liyc5929/neuroseqbench.git
```

Then from this folder:

```
python train.py                 # trains and saves model_data/har_lif_ff.pt
python benchmark.py             # NeuroBench metrics on the test split
```

For the recurrent variant use `python train.py --recurrent` and
`python benchmark.py --checkpoint model_data/har_lif_rec.pt`.

## Metrics

`benchmark.py` reports ParameterCount, Footprint, ConnectionSparsity,
ClassificationAccuracy, ActivationSparsity and SynapticOperations. NSA's
`SpikeGeneration` module is registered as an activation module so the spiking layers
are hooked. Synaptic operations count every one of the 200 time steps. Only the first
layer sees real-valued input (MACs), every later layer sees spikes (ACs).

## Results

Both models trained with `train.py` defaults (seed 1234) on an RTX 4060 Ti, then
evaluated with `benchmark.py` on NSA's test split (6,512 windows).

| Model | Accuracy | NSA paper | Parameters | Activation sparsity | Effective MACs | Effective ACs | Dense SynOps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LIF feedforward | 81.62% | 81.27% | 103,954 | 0.803 | 76,800 | 2,528,114 | 20,659,200 |
| LIF recurrent | 74.11% | 77.32% | 105,218 | 0.671 | 76,800 | 5,316,493 | 20,883,200 |

SynOps are per sample, summed over all 200 time steps, and include the recurrent
weights. Effective operations (MACs plus ACs) come to about 12.6% of the dense count for
the feedforward model and 25.8% for the recurrent one.

The feedforward model matches the paper. The recurrent model uses the hyperparameters
from NSA's `HAR/run_all.sh` but lands about 3 points below the paper on a single seed.
Seed variance or a setting not captured in that script are the likely causes, and more
seeds are needed to tell which. The NSA paper column is Table 3 of
[arXiv:2505.22035](https://arxiv.org/abs/2505.22035).

The trained checkpoints are in `model_data/` (`har_lif_ff.pt`, `har_lif_rec.pt`), so
`benchmark.py` reproduces both rows without retraining.
