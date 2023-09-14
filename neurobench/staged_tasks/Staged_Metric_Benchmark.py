from tqdm import tqdm
from neurobench.benchmarks.benchmark import Benchmark
import importlib

from neurobench.benchmarks import static_metrics, data_metrics


class Staged_Metric_Benchmark(Benchmark):
    """ Top0-level benchmark class for running benchmarks on staged tasks.
    """
    def __init__(self, model, staged_dataloader, staged_preprocessors, staged_postprocessors, metric_list, staged_metrics_location):
        super().__init__(model, staged_dataloader, staged_preprocessors, staged_postprocessors, metric_list)
        """
        Args:
            staged_model: A NeuroBenchModel.
            staged_dataloader: A PyTorch DataLoader.
            staged_preprocessors: A list of NeuroBenchProcessors.
            staged_postprocessors: A list of NeuroBenchAccumulators.
            metric_list: A list of lists of strings of metrics to run. 
                First item is static metrics, second item is data metrics, third item is the staged metrics.
            staged_metrics_location: The relative path of the file with the staged metrics."""
        staged_module = importlib.import_module(staged_metrics_location)
        self.staged_metrics = {m: getattr(staged_module, m) for m in metric_list[3]}

    def run_staged(self):
        """ Runs batched evaluation of the benchmark.
        """
        print("Running staging benchmark")

        # Static metrics
        results = {}
        for m in self.static_metrics.keys():
            results[m] = self.static_metrics[m](self.model)

        # Init/re-init stateful data metrics
        for m in self.data_metrics.keys():
            if isinstance(self.data_metrics[m],type) and issubclass(self.data_metrics[m], data_metrics.AccumulatedMetric):
                self.data_metrics[m] = self.data_metrics[m]()
            elif isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                self.data_metrics[m] = self.data_metrics[m]()

        # Init/re-init stateful staged metrics
        for m in self.staged_metrics.keys():
            if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                self.data_metrics[m] = self.data_metrics[m]()

        dataset_len = len(self.dataloader.dataset)
        for data in tqdm(self.dataloader, total=len(self.dataloader)):
            batch_size = data[0].size(0)

            # convert data to tuple
            if type(data) is not tuple:
                data = tuple(data)

            # Preprocessing data
            for alg in self.preprocessors:
                data = alg(data)

            # Run model on test data
            preds = self.model(data[0])

            for alg in self.postprocessors: 
                preds = alg(preds)

            # Data metrics
            batch_results = {}
            for m in self.data_metrics.keys():
                batch_results[m] = self.data_metrics[m](self.model, preds, data)
            
            
            for m, v in batch_results.items():
                # AccumulatedMetrics are computed with past state
                if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                    results[m] = v
                # otherwise accumulate via mean
                else:
                    assert isinstance(v, float) or isinstance(v, int), "Data metric must return float or int to be accumulated"
                    if m not in results:
                        results[m] = v * batch_size / dataset_len
                    else:
                        results[m] += v * batch_size / dataset_len
            
            # Staged metrics
            staged_results = {}
            for m in self.staged_metrics.keys():
                staged_results[m] = self.staged_metrics[m](self.model, preds, data)

            for m, v in staged_results.items():
                # AccumulatedMetrics are computed with past state
                if isinstance(self.staged_metrics[m], data_metrics.AccumulatedMetric):
                    results[m] = v
                # otherwise accumulate via mean
                else:
                    assert isinstance(v, float) or isinstance(v, int), "Staged metric must return float or int to be accumulated"
                    if m not in results:
                        results[m] = v * batch_size / dataset_len
                    else:
                        results[m] += v * batch_size / dataset_len

        return results

    

