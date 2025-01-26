Mackey Glass Prediction Benchmark
=======================================================================
The Mackey Glass prediction benchmark task tests a model's autoregressive prediction capability of a chaotic system.
More information is available in the MG_tutorial notebook and associated scripts `esn_benchmark.py` and `lstm_benchmark.py`.
The task script and infrastructure were built by Denis Kleyko and Younes Bouhadjar.

## Notice
Due to the underlying arithmetic solvers used in integration to produce the Mackey Glass time series, the dataset should be downloaded and used as-is, rather than reproduced on a different machine. The associated MackeyGlass NeuroBench dataset should automatically download the data.