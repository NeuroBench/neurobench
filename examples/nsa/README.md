# Neuromorphic Sequential Arena (NSA) baselines

Examples that run baselines from the
[Neuromorphic Sequential Arena](https://arxiv.org/abs/2505.22035) (Ma et al., IJCAI
2025) through NeuroBench, following up
[issue #264](https://github.com/NeuroBench/neurobench/issues/264).

NSA's models live in the external
[`neuroseqbench`](https://github.com/liyc5929/neuroseqbench) package. These examples
import it rather than copying any of its code into NeuroBench.

| Task | Folder | Status |
| --- | --- | --- |
| Human Activity Recognition (HAR) | [`har/`](har/) | LIF baselines |
| Autonomous Localization (AL) | | planned |
| EEG Motor Imagery (EEG-MI) | | planned |
| Sound Source Localization (SSL) | | planned |

## Shared helpers

`nsa_common.py` holds the glue every task uses:

- `TimeFirstWrapper` turns NeuroBench's batch-first `[B, T, C]` input into NSA's
  time-first `[T, B, C]`, and applies NSA's readout (mean over time, or the last step for
  EEG-MI).
- `wrap_for_neurobench` wraps the network as a `TorchModel` and registers NSA's
  `SpikeGeneration` modules as activations, so ActivationSparsity and
  SynapticOperations see the spikes.

## Tests

`tests/test_nsa_examples.py` covers the wrapper, hook detection and the metric values.
It is skipped when `neuroseqbench` is not installed.
