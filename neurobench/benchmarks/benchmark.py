import sys
from contextlib import redirect_stdout
from tqdm import tqdm

from . import static_metrics, workload_metrics

# workload metrics which require hooks
requires_hooks = [
    "activation_sparsity",
    "number_neuron_updates",
    "synaptic_operations",
    "membrane_updates",
]


class Benchmark:
    """Top-level benchmark class for running benchmarks."""

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

        self.static_metrics = {m: getattr(static_metrics, m) for m in metric_list[0]}
        self.workload_metrics = {
            m: getattr(workload_metrics, m) for m in metric_list[1]
        }

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
            results = {}
            for m in self.static_metrics.keys():
                results[m] = self.static_metrics[m](self.model)

            # add hooks to the model
            if any([m in requires_hooks for m in self.workload_metrics.keys()]):
                workload_metrics.detect_activations_connections(self.model)

            dataloader = dataloader if dataloader is not None else self.dataloader
            preprocessors = (
                preprocessors if preprocessors is not None else self.preprocessors
            )
            postprocessors = (
                postprocessors if postprocessors is not None else self.postprocessors
            )

            # Init/re-init stateful data metrics
            for m in self.workload_metrics.keys():
                if isinstance(self.workload_metrics[m], type) and issubclass(
                    self.workload_metrics[m], workload_metrics.AccumulatedMetric
                ):
                    self.workload_metrics[m] = self.workload_metrics[m]()
                elif isinstance(
                    self.workload_metrics[m], workload_metrics.AccumulatedMetric
                ):  # new benchmark run, reset metric state
                    self.workload_metrics[m].reset()

            dataset_len = len(dataloader.dataset)

            if device is not None:
                self.model.net.to(device)

            batch_num = 0
            for data in tqdm(dataloader, total=len(dataloader), disable=quiet):
                if device is not None:
                    data = (data[0].to(device), data[1].to(device))

                batch_size = data[0].size(0)

                # convert data to tuple
                if type(data) is not tuple:
                    data = tuple(data)

                # Preprocessing data
                for alg in preprocessors:
                    data = alg(data)

                # Run model on test data
                preds = self.model(data[0])

                for alg in postprocessors:
                    preds = alg(preds)

                # Data metrics
                batch_results = {}
                for m in self.workload_metrics.keys():
                    batch_results[m] = self.workload_metrics[m](self.model, preds, data)

                for m, v in batch_results.items():
                    # AccumulatedMetrics are computed after all batches complete
                    if isinstance(
                        self.workload_metrics[m], workload_metrics.AccumulatedMetric
                    ):
                        continue
                    # otherwise accumulate via mean
                    else:
                        assert isinstance(v, float) or isinstance(
                            v, int
                        ), "Data metric must return float or int to be accumulated"
                        if m not in results:
                            results[m] = v * batch_size / dataset_len
                        else:
                            results[m] += v * batch_size / dataset_len

                # delete hook contents
                self.model.reset_hooks()

                if verbose:
                    for m in self.workload_metrics.keys():
                        if isinstance(
                            self.workload_metrics[m], workload_metrics.AccumulatedMetric
                        ):
                            results[m] = self.workload_metrics[m].compute()
                    print(f"\nBatch num {batch_num + 1}/{len(dataloader)}")
                    print(results)

                batch_num += 1

            # compute AccumulatedMetrics after all batches if they are not calculated at every iteration
            if not verbose:
                for m in self.workload_metrics.keys():
                    if isinstance(
                        self.workload_metrics[m], workload_metrics.AccumulatedMetric
                    ):
                        results[m] = self.workload_metrics[m].compute()

        # close hooks
        for hook in self.model.activation_hooks:
            hook.reset()
            hook.close()
        for hook in self.model.connection_hooks:
            hook.reset()
            hook.close()
        self.model.activation_hooks = []
        self.model.connection_hooks = []

        return results
