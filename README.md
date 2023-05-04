# algorithms_benchmarks

Benchmark harness and baseline results for the NeuroBench algorithms track.

## API
A proposed API structure and a few of the many outstanding questions.

`neurobench.benchmarks`: Benchmark pipelines that take a `Dataset`, `Model`, & list of `PreProcessor` and output `Result`
- Metrics management?
- Model simulation?
- Hardware deployment?
- Result publishing?

`neurobench.datasets`: Wrappers for existing datasets, focusing on consistency and performant loading
- Dataset downloading?
- Dataset folder structure?
- Storage of pre-processed intermediate data?

`neurobench.models`: Light-weight wrapper for existing model frameworks, focusing on support for existing SNN frameworks (SNNTorch, Rockpool, etc.), typical ANN models, and forcing a consistent data dimensionality/structure between them (eg. `[batch, ..., timesteps]`)
- Inference API?

`neurobench.preprocessing`: Functions for processing data to prepare it for neuromorphic models (spike conversion algorithms, etc.)
- Benchmarking preprocessors?
