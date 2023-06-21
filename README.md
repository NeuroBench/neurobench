# algorithms_benchmarks
Benchmark harness and baseline results for the NeuroBench algorithms track.

## API
This branch contains an example API that highlights how pieces of NeuroBench might be implemented. It is not meant to be a definitive solution, in certain cases (like running processors and metrics on data) two different approaches are used to show variety in implementation.

It also shows how dependencies can be managed with optional installations, preventing unnecessary bloat. Optional dependencies are specified using the `-E` flag when running `poetry install` or placed in square brackets when installing the final wheel from PyPI (`pip install neurobench[snntorch]`).

## Running the code
To run the code, poetry can be used to maintain a virtualenv consistent with a deployment environment. In the `algorithms_benchmarks` folder run:
```
poetry install -E "snntorch speech2spikes"
poetry run python neurobench/examples/benchmark_model.py
```