import sys
from contextlib import redirect_stdout
from os import mkdir
from tqdm import tqdm
from snntorch import utils
from neurobench.metrics.manager.static_manager import StaticMetricManager
from neurobench.metrics.manager.workload_manager import WorkloadMetricManager
from neurobench.processors.manager import ProcessorManager
from neurobench.models import NeuroBenchModel, SNNTorchModel, SNNTorchAgent, TorchModel
from torch.utils.data import DataLoader
from neurobench.processors.abstract import (
    NeuroBenchPreProcessor,
    NeuroBenchPostProcessor,
)
from neurobench.metrics.abstract import StaticMetric, WorkloadMetric
import json
import csv
import os
from typing import Literal, List, Type, Optional, Dict, Any, Callable, Tuple, Union
import pathlib
import snntorch
from torch import Tensor
import torch


class BenchmarkClosedLoop:
    """Top-level benchmark class for running closed loop benchmarks."""

    def __del__(self):
        if hasattr(self, "workload_metric_manager"):
            self.workload_metric_manager.cleanup_hooks(self.agent)

    def __init__(
        self,
        agent: Union[SNNTorchAgent, TorchModel],
        environment,
        preprocessors: Optional[
            List[
                NeuroBenchPreProcessor
                | Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
            ]
        ],
        postprocessors: Optional[
            List[NeuroBenchPostProcessor | Callable[[Tensor], Tensor]]
        ],
        metric_list: List[List[Type[StaticMetric | WorkloadMetric]]],
    ):
        """
        Args:
            agent: A NeuroBenchAgent (SNNTorchAgent/TorchModel).
            environment: A Gym environment.
            preprocessors: A list of NeuroBenchPreProcessors or callable functions (e.g. lambda) with matching interfaces.
            postprocessors: A list of NeuroBenchPostProcessors or callable functions (e.g. lambda) with matching interfaces.
            metric_list: A list of lists of StaticMetric and WorkloadMetric classes of metrics to run.
                First item is StaticMetrics, second item is WorkloadMetrics.
        """
        self.agent = agent
        self.env = environment  # env not dataset
        self.processor_manager = ProcessorManager(preprocessors, postprocessors)
        self.static_metric_manager = StaticMetricManager(metric_list[0])
        self.workload_metric_manager = WorkloadMetricManager(metric_list[1])
        self.workload_metric_manager.register_hooks(agent)
        self.results = None

    def run(
        self,
        quiet: bool = False,
        verbose: bool = False,
        nr_interactions=100,
        max_length=1000,
        device: Optional[str] = None,
    ):
        """
        Runs batched evaluation of the benchmark.

        Currently, data metrics are accumulated via mean over the entire
        test set, and thus must return a float or int.
        Args:
            nr_interactions: Number of interactions with the environment.
            max_length: Maximum length of an interaction with the environment.
        Returns:
            results: A dictionary of results.

        """
        with redirect_stdout(None if quiet else sys.stdout):
            print("Running benchmark")

            self.results = None
            results = self.static_metric_manager.run_metrics(self.agent)

            # No preprocessors and postprocessors needed for the environment or agent.
            # if preprocessors is not None:
            #     self.processor_manager.replace_preprocessors(preprocessors)
            # if postprocessors is not None:
            #     self.processor_manager.replace_postprocessors(postprocessors)

            self.workload_metric_manager.initialize_metrics()

            if device is not None:
                self.agent.__net__().to(device)

            batch_num = 0
            successful_trials = 0
            rewards = []
            time_taken = []

            with torch.no_grad():
                for _ in tqdm(range(nr_interactions)):
                    env = self.env
                    # self.agent.reset() # Is this needed?

                    # get initial state
                    state, _ = env.reset()
                    # state = env.set_state(constant_state)

                    if device is not None:
                        state = state.to(device)

                    t_sim = (
                        0  # Might not need this as our environment keep track of time
                    )
                    reward_tot = 0
                    times = []
                    terminal = False

                    while not terminal and t_sim < max_length:
                        # Preprocessing data
                        # input, target = self.processor_manager.preprocess(state) # Check if this is needed

                        # get network outputs on given state
                        output = self.agent(state.unsqueeze(0).unsqueeze(0))

                        # Postprocessing data
                        # output = self.processor_manager.postprocess(output)

                        # perform action
                        obs, reward, terminal, _, _ = env.step(
                            output
                        )  # Do we need to include reward here, as this is used for benchmarking, not training?

                        reward_tot += reward
                        if not terminal:
                            state = obs

                        t_sim += 1
                        times.append(t_sim)
                    rewards.append(reward_tot)
                    time_taken.append(t_sim*env.ops.time_step)
                    if env.time_in_range * env.ops.time_step >= env.min_time_in_target:
                        successful_trials += 1

                    # Data metrics
                    # Predictions and data handled by environment.
                    # Dummy tensors used for metrics management.
                    preds = torch.tensor([1])
                    data = (torch.tensor([1]), torch.tensor([1]))

                    batch_results = self.workload_metric_manager.run_metrics(
                        self.agent,
                        preds=preds,
                        data=data,
                        batch_size=1,
                        dataset_len=nr_interactions,
                    )
                    self.workload_metric_manager.reset_hooks(self.agent)

                    if verbose:
                        results.update(batch_results)
                        print(f"\nBatch num {batch_num + 1}/{len(dataloader)}")
                        print(dict(results))

                    # delete hook contents
                    self.agent.reset_hooks()  # Is this still necessary with the line self.workload_metric_manager.reset_hooks(self.model)

                    batch_num += 1

                results.update(self.workload_metric_manager.results)
                self.workload_metric_manager.clean_results()
                self.results = dict(results)
            average_time_taken = sum(time_taken) / len(time_taken)
            print(f"Successful trials: {successful_trials}/{nr_interactions}")
            print(f"Average time taken: {average_time_taken:.2f} seconds")

        return self.results, average_time_taken
