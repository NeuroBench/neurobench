import sys
from contextlib import redirect_stdout
from tqdm import tqdm

from . import static_metrics, data_metrics

class Benchmark():
    """ Top-level benchmark class for running benchmarks.
    """
    def __init__(self, model, dataloader, preprocessors, postprocessors, metric_list):
        """
        Args:
            model: A NeuroBenchModel.
            dataloader: A PyTorch DataLoader.
            preprocessors: A list of NeuroBenchProcessors.
            postprocessors: A list of NeuroBenchAccumulators.
            metric_list: A list of lists of strings of metrics to run. 
                First item is static metrics, second item is data metrics.
        """

        self.model = model 
        self.dataloader = dataloader # dataloader not dataset
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

        self.static_metrics = {m: getattr(static_metrics, m) for m in metric_list[0]}
        self.data_metrics = {m: getattr(data_metrics, m) for m in metric_list[1]}

    def run(self, quiet=False, verbose: bool = False, 
            dataloader=None, preprocessors=None, postprocessors=None):
        """ Runs batched evaluation of the benchmark.

        Args:
            dataloader (optional): override DataLoader for this run.
            preprocessors (optional): override preprocessors for this run.
            postprocessors (optional): override postprocessors for this run.
            quiet (bool, default=False): If True, output is suppressed.
            verbose (bool, default=False): If True, metrics for each bach will be printed.
                                           If False (default), metrics are accumulated and printed after all batches are processed.

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
            data_metrics.detect_activations_connections(self.model)

            dataloader = dataloader if dataloader is not None else self.dataloader
            preprocessors = preprocessors if preprocessors is not None else self.preprocessors
            postprocessors = postprocessors if postprocessors is not None else self.postprocessors

            # Init/re-init stateful data metrics
            for m in self.data_metrics.keys():
                if isinstance(self.data_metrics[m],type) and issubclass(self.data_metrics[m], data_metrics.AccumulatedMetric):
                    self.data_metrics[m] = self.data_metrics[m]()
                elif isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric): # new benchmark run, reset metric state
                    self.data_metrics[m].reset()

            dataset_len = len(dataloader.dataset)

            batch_num = 0
            for data in tqdm(dataloader, total=len(dataloader), disable=quiet):
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
                for m in self.data_metrics.keys():
                    batch_results[m] = self.data_metrics[m](self.model, preds, data)

                for m, v in batch_results.items():
                    # AccumulatedMetrics are computed after all batches complete
                    if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                        continue
                    # otherwise accumulate via mean
                    else:
                        assert isinstance(v, float) or isinstance(v, int), "Data metric must return float or int to be accumulated"
                        if m not in results:
                            results[m] = v * batch_size / dataset_len
                        else:
                            results[m] += v * batch_size / dataset_len

                # delete hook contents
                self.model.reset_hooks()

                if verbose:
                    for m in self.data_metrics.keys():
                        if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                            results[m] = self.data_metrics[m].compute()
                    print(f'\nBatch num {batch_num + 1}/{len(dataloader)}')
                    print(results)

                batch_num += 1

            # compute AccumulatedMetrics after all batches if they are not calculated at every iteration
            if not verbose:
                for m in self.data_metrics.keys():
                    if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                        results[m] = self.data_metrics[m].compute()

        return results
