from tqdm import tqdm
from . import metrics

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

        self.static_metrics = {m: getattr(metrics, m) for m in metric_list[0]}
        self.data_metrics = {m: getattr(metrics, m) for m in metric_list[1]}

    def run(self):
        """ Runs batched evaluation of the benchmark.

        Currently, data metrics are accumulated via mean over the entire
        test set, and thus must return a float or int.

        Returns:
            results: A dictionary of results.
        """
        print("Running benchmark")

        # Static metrics
        results = {}
        for m in self.static_metrics.keys():
            results[m] = self.static_metrics[m](self.model)

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

            # TODO: postprocessors are applied to model output only?
            for alg in self.postprocessors: 
                preds = alg(preds)

            # Data metrics
            batch_results = {}
            for m in self.data_metrics.keys():
                batch_results[m] = self.data_metrics[m](self.model, preds, data)

            # Accumulate data metrics via mean
            for m, v in batch_results.items():
                assert isinstance(v, float) or isinstance(v, int), "Data metric must return float or int to be accumulated"
                if m not in results:
                    results[m] = v * batch_size / dataset_len
                else:
                    results[m] += v * batch_size / dataset_len

        return results
