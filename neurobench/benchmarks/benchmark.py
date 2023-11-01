from tqdm import tqdm

from . import static_metrics, data_metrics

import torch
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

    def run(self):
        """ Runs batched evaluation of the benchmark.

        Currently, data metrics are accumulated via mean over the entire
        test set, and thus must return a float or int.

        Returns:
            results: A dictionary of results.
        """
        print("Running benchmark")
        
        # add hooks to the model
        data_metrics.detect_activations_connections(self.model)

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

        dataset_len = len(self.dataloader.dataset)
        
        batch_num = 0
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

            batch_num += 1
                

        # compute AccumulatedMetrics after all batches
        for m in self.data_metrics.keys():
            if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                results[m] = self.data_metrics[m].compute()

        return results



class Benchmark_Closed_Loop():
    """ Top-level benchmark class for running benchmarks.
    """
    def __init__(self, agent, environment, preprocessors, postprocessors, metric_list):
        """
        Args:
            agent: A NeuroBenchAgent (SNNTorchAgent/TorchModel).
            environment: A Gym environment.
            preprocessors: A list of NeuroBenchProcessors.
            postprocessors: A list of NeuroBenchAccumulators.
            metric_list: A list of lists of strings of metrics to run. 
                First item is static metrics, second item is data metrics.
        """
        self.agent = agent
        self.env = environment # env not dataset
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

        self.static_metrics = {m: getattr(static_metrics, m) for m in metric_list[0]}
        self.data_metrics = {m: getattr(data_metrics, m) for m in metric_list[1]}

    def run(self, nr_interactions = 100, max_length = 1000):
        """ Runs batched evaluation of the benchmark.

        Currently, data metrics are accumulated via mean over the entire
        test set, and thus must return a float or int.
        Args:
            nr_interactions: Number of interactions with the environment.
            max_length: Maximum length of an interaction with the environment.
        Returns:
            results: A dictionary of results.
        """
        print("Running benchmark")
        
        # add hooks to the agent
        data_metrics.detect_activations_connections(self.agent)

        # Static metrics
        results = {}
        for m in self.static_metrics.keys():
            results[m] = self.static_metrics[m](self.agent)

        # Init/re-init stateful data metrics
        for m in self.data_metrics.keys():
            if isinstance(self.data_metrics[m],type) and issubclass(self.data_metrics[m], data_metrics.AccumulatedMetric):
                self.data_metrics[m] = self.data_metrics[m]()
            elif isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                self.data_metrics[m] = self.data_metrics[m]()

        dataset_len = nr_interactions
        
        batch_num = 0
        reward = []
        
        for _ in tqdm(range(nr_interactions)):
            env = self.env
            self.agent.reset() 
            
            # get initial state
            state, _ = env.reset()
            # state = env.set_state(constant_state)

            t_sim = 0
            rewards = []
            times = []
            terminal = False

            # print('Interacting with environment')
            while not terminal and t_sim < max_length:

                # state to tensor
                state = torch.from_numpy(state)

                # Preprocessing state
                for alg in self.preprocessors:
                    state = alg(state)

                # get network outputs on given state
                output = self.agent(state.unsqueeze(0))


                # Postprocessing output to get action
                for alg in self.postprocessors:
                    output = alg(output)

                # perform action
                obs, reward, terminal, _, _ = env.step(output)


                if not terminal:
                    state = obs

                t_sim += 1
            rewards.append(reward)
            times.append(t_sim)
            reward = reward

            # Data metrics
            # we need a data term to compute the synaptic operations, only necessary for averaging
            data = torch.tensor([rewards,times]).unsqueeze(0)
            batch_results = {}
            for m in self.data_metrics.keys():
                batch_results[m] = self.data_metrics[m](self.agent, reward, t_sim)

            for m, v in batch_results.items():
                # AccumulatedMetrics are computed after all batches complete
                if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                    continue
                # otherwise accumulate via mean
                else:
                    assert isinstance(v, float) or isinstance(v, int), "Data metric must return float or int to be accumulated"
                    if m not in results:
                        results[m] = v / dataset_len
                    else:
                        results[m] += v / dataset_len
            
            # delete hook contents
            self.agent.reset_hooks()

            batch_num += 1
                

        # compute AccumulatedMetrics after all batches
        for m in self.data_metrics.keys():
            if isinstance(self.data_metrics[m], data_metrics.AccumulatedMetric):
                results[m] = self.data_metrics[m].compute()

        return results


