"""
"""
from tqdm import tqdm
from . import metrics

class Benchmark():
    def __init__(self, model, data, processors, metric_list):
        self.model = model
        self.data = data
        self.processors = processors
        self.metrics = {m: getattr(metrics, m) for m in metric_list}

    def run(self):
        print("Preprocessing data")
        data = self.data
        for alg in self.processors:
            data = zip(*alg(tqdm(data)))

        print("Running model on test data")
        preds = []
        for d in tqdm(data, total=len(self.data)):
            preds.append(self.model(d[0]))
        
        print("Calculating metrics")
        results = {}
        for m in tqdm(self.metrics):
            results[m] = self.metrics[m](self.model, self.data, preds)

        return results
