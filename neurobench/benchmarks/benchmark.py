"""
"""
from tqdm import tqdm
from . import metrics

class Benchmark():
    def __init__(self, model, dataloader, preprocessors, accumulators, metric_list):
        self.model = model
        self.dataloader = dataloader # assuming dataloader not dataset
        self.preprocessors = preprocessors
        self.accumulators = accumulators

        self.static_metrics = {m: getattr(metrics, m) for m in metric_list[0]}
        self.data_metrics = {m: getattr(metrics, m) for m in metric_list[1]}

    def run(self):
        print("Running benchmark")

        # Static metrics
        results = {}
        for m in self.static_metrics.keys():
            results[m] = self.static_metrics[m](self.model)

        dataset_len = len(self.dataloader.dataset)
        for data in tqdm(self.dataloader, total=len(self.dataloader)):
            batch_size = data[0].size(0)

            # Preprocessing data
            for alg in self.preprocessors:
                data = zip(*alg(data))

            # Run model on test data
            preds = self.model(data[0])

            for alg in self.accumulators: 
                preds = alg(preds)

            # Data metrics
            batch_results = {}
            for m in self.data_metrics.keys():
                batch_results[m] = self.data_metrics[m](self.model, preds, data)

            # Accumulate data metrics via mean
            for m, v in batch_results.items(): # TODO: add check for v to be float/int
                assert isinstance(v, float) or isinstance(v, int), "Data metric must return float or int to be accumulated"
                if m not in results:
                    results[m] = v * batch_size / dataset_len
                else:
                    results[m] += v * batch_size / dataset_len

        return results
