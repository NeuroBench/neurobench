"""
"""
from tqdm import tqdm
from . import metrics

class FSCILBenchmark():
    def __init__(self, model, dataloader, processors, metric_list):
        self.model = model
        self.dataloader = iter(dataloader)
        self.processors = processors
        self.metrics = {m: getattr(metrics, m) for m in metric_list}

    def run(self):
        try:
            data = next(self.dataloader)
        except StopIteration:
            raise ValueError("Test dataloader is empty")

        run_data = {}
        run_data["model"] = self.model
        run_data["data"] = self.data

        print("Preprocessing data")
        data = self.data
        for alg in self.processors:
            data = zip(*alg(tqdm(data)))

        print("Running model on test data")
        run_data["preds"] = []
        for d in tqdm(data, total=len(self.data)):
            pred = self.model(d[0]) # TODO: prediction is output to be compared to labels?
            run_data["preds"].append(pred)

            batch_data = model.track_batch()
            for k, v in batch_data.items():
                if k not in run_data:
                    run_data[k] = []
                run_data[k].append(v)

        run_data = run_data | model.track_run()
        
        print("Calculating metrics")
        results = {}
        for m in tqdm(self.metrics):
            results[m] = self.metrics[m](run_data)

        return results