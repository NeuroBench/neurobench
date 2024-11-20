import sys
from contextlib import redirect_stdout
from tqdm import tqdm
from neurobench.metrics.manager.static_manager import StaticMetricManager
from neurobench.metrics.manager.workload_manager import WorkloadMetricManager


class Benchmark:
    """Top-level benchmark class for running benchmarks."""

    def __del__(self):
        self.workload_metric_orchestrator.cleanup_hooks(self.model)

    def __init__(self, model, dataloader, preprocessors, postprocessors, metric_list):
        """
        Args:
            model: A NeuroBenchModel.
            dataloader: A PyTorch DataLoader.
            preprocessors: A list of NeuroBenchPreProcessors.
            postprocessors: A list of NeuroBenchPostProcessors.
            metric_list: A list of lists of strings of metrics to run.
                First item is static metrics, second item is data metrics.
        """

        self.model = model
        self.dataloader = dataloader  # dataloader not dataset
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

        self.static_metric_orchestrator = StaticMetricManager(metric_list[0])
        self.workload_metric_orchestrator = WorkloadMetricManager(metric_list[1])

        self.workload_metric_orchestrator.register_hooks(model)

    def run(
        self,
        quiet=False,
        verbose: bool = False,
        dataloader=None,
        preprocessors=None,
        postprocessors=None,
        device=None,
    ):
        """
        Runs batched evaluation of the benchmark.

        Args:
            dataloader (optional): override DataLoader for this run.
            preprocessors (optional): override preprocessors for this run.
            postprocessors (optional): override postprocessors for this run.
            quiet (bool, default=False): If True, output is suppressed.
            verbose (bool, default=False): If True, metrics for each bach will be printed.
                                           If False (default), metrics are accumulated and printed after all batches are processed.
            device (optional): use device for this run (e.g. 'cuda' or 'cpu').

        Returns:
            results: A dictionary of results.

        """
        with redirect_stdout(None if quiet else sys.stdout):
            print("Running benchmark")

            # Static metrics
            results = self.static_metric_orchestrator.run_metrics(self.model)

            dataloader = dataloader if dataloader is not None else self.dataloader
            preprocessors = (
                preprocessors if preprocessors is not None else self.preprocessors
            )
            postprocessors = (
                postprocessors if postprocessors is not None else self.postprocessors
            )

            self.workload_metric_orchestrator.initialize_metrics()

            dataset_len = len(dataloader.dataset)

            if device is not None:
                self.model.net.to(device)

            batch_num = 0
            for data in tqdm(dataloader, total=len(dataloader), disable=quiet):
                # convert data to tuple
                data = tuple(data) if not isinstance(data, tuple) else data

                if device is not None:
                    data = (data[0].to(device), data[1].to(device))

                batch_size = data[0].size(0)

                # Preprocessing data
                for alg in preprocessors:
                    data = alg(data)

                # Run model on test data
                preds = self.model(data[0])

                for alg in postprocessors:
                    preds = alg(preds)

                # Data metrics
                batch_results = self.workload_metric_orchestrator.run_metrics(
                    self.model, preds, data, batch_size, dataset_len
                )
                self.workload_metric_orchestrator.reset_hooks(self.model)

                if verbose:
                    results.update(batch_results)
                    print(f"\nBatch num {batch_num + 1}/{len(dataloader)}")
                    print(dict(results))

                batch_num += 1

            results.update(self.workload_metric_orchestrator.results)
            self.workload_metric_orchestrator.clean_results()

        return dict(results)
