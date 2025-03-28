import sys
from contextlib import redirect_stdout
from os import mkdir
from tqdm import tqdm
from snntorch import utils
from neurobench.metrics.manager.static_manager import StaticMetricManager
from neurobench.metrics.manager.workload_manager import WorkloadMetricManager
from neurobench.processors.manager import ProcessorManager
from neurobench.models import NeuroBenchModel, SNNTorchModel
from torch.utils.data import DataLoader
from neurobench.processors.abstract import (
    NeuroBenchPreProcessor,
    NeuroBenchPostProcessor,
)
from neurobench.metrics.abstract import StaticMetric, WorkloadMetric
import json
import csv
import os
from typing import Literal, List, Type, Optional, Dict, Any, Callable, Tuple
import pathlib
import snntorch
from torch import Tensor
import torch


class Benchmark:
    """Top-level benchmark class for running benchmarks."""

    def __del__(self):
        if hasattr(self, "workload_metric_manager"):
            self.workload_metric_manager.cleanup_hooks(self.model)

    def __init__(
        self,
        model: NeuroBenchModel,
        dataloader: Optional[DataLoader],
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
            model: A NeuroBenchModel.
            dataloader: A PyTorch DataLoader.
            preprocessors: A list of NeuroBenchPreProcessors or callable functions (e.g. lambda) with matching interfaces.
            postprocessors: A list of NeuroBenchPostProcessors or callable functions (e.g. lambda) with matching interfaces.
            metric_list: A list of lists of StaticMetric and WorkloadMetric classes of metrics to run.
                First item is StaticMetrics, second item is WorkloadMetrics.
        """

        self.model = model
        self.dataloader = dataloader  # dataloader not dataset
        self.processor_manager = ProcessorManager(preprocessors, postprocessors)
        self.static_metric_manager = StaticMetricManager(metric_list[0])
        self.workload_metric_manager = WorkloadMetricManager(metric_list[1])
        self.workload_metric_manager.register_hooks(model)
        self.results = None

    def run(
        self,
        quiet: bool = False,
        verbose: bool = False,
        dataloader: Optional[DataLoader] = None,
        preprocessors: Optional[
            NeuroBenchPreProcessor
            | Callable[[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]
        ] = None,
        postprocessors: Optional[
            NeuroBenchPostProcessor | Callable[[Tensor], Tensor]
        ] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs batched evaluation of the benchmark.

        Args:
            dataloader (Optional): override DataLoader for this run.
            preprocessors (Optional): override preprocessors for this run.
            postprocessors (Optional): override postprocessors for this run.
            quiet (bool, default=False): If True, output is suppressed.
            verbose (bool, default=False): If True, metrics for each bach will be printed.
                                           If False (default), metrics are accumulated and printed after all batches are processed.
            device (Optional): use device for this run (e.g. 'cuda' or 'cpu').

        Returns:
            Dict[str, Any]: A dictionary of results.

        """
        with redirect_stdout(None if quiet else sys.stdout):
            print("Running benchmark")

            self.results = None
            results = self.static_metric_manager.run_metrics(self.model)

            dataloader = dataloader if dataloader is not None else self.dataloader

            if preprocessors is not None:
                self.processor_manager.replace_preprocessors(preprocessors)
            if postprocessors is not None:
                self.processor_manager.replace_postprocessors(postprocessors)

            self.workload_metric_manager.initialize_metrics()

            dataset_len = len(dataloader.dataset)

            if device is not None:
                self.model.__net__().to(device)

            batch_num = 0
            for data in tqdm(dataloader, total=len(dataloader), disable=quiet):
                # convert data to tuple
                data = tuple(data) if not isinstance(data, tuple) else data

                if device is not None:
                    data = (data[0].to(device), data[1].to(device))

                batch_size = data[0].size(0)

                # Preprocessing data
                input, target = self.processor_manager.preprocess(data)

                # Run model on test data
                preds = self.model(input)

                # Postprocessing data
                preds = self.processor_manager.postprocess(preds)

                # Data metrics
                batch_results = self.workload_metric_manager.run_metrics(
                    self.model, preds, data, batch_size, dataset_len
                )
                self.workload_metric_manager.reset_hooks(self.model)

                if verbose:
                    results.update(batch_results)
                    print(f"\nBatch num {batch_num + 1}/{len(dataloader)}")
                    print(dict(results))

                batch_num += 1

            results.update(self.workload_metric_manager.results)
            self.workload_metric_manager.clean_results()
            self.results = dict(results)
        return self.results

    def save_benchmark_results(
        self, file_path: str, file_format: Literal["json", "csv", "txt"] = "json"
    ) -> None:
        """
        Save benchmark results to a specified file in the chosen format.

        Args:
            file_path (str):
                Path to the output file (excluding the extension). The method
                automatically appends the appropriate extension based on the
                chosen file format.
            file_format (Literal["json", "csv", "txt"], default="json"):
                The format in which the results should be saved. Supported formats:

                - `"json"`: Saves the results as a JSON file with formatted indentation.

                - `"csv"`: Saves the results as a CSV file with keys as headers and values as the first row.

                - `"txt"`: Saves the results as a plain text file with one key-value pair per line.

        Raises:
            ValueError:
                If the provided `file_format` is not one of the supported formats (`"json"`, `"csv"`, `"txt"`).

        """
        file_format = file_format.lower()

        # Ensure the directory exists
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)

        # JSON format
        if file_format == "json":
            with open(f"{file_path}.json", "w") as json_file:
                json.dump(self.results, json_file, indent=4)
            print(f"Results saved to {file_path}.json")

        # CSV format
        elif file_format == "csv":
            with open(f"{file_path}.csv", "w", newline="") as csv_file:
                writer = csv.writer(csv_file)

                # Write header (keys)
                writer.writerow(self.results.keys())

                # Write values
                writer.writerow(self.results.values())
            print(f"Results saved to {file_path}.csv")

        # Plain Text format
        elif file_format == "txt":
            with open(f"{file_path}.txt", "w") as txt_file:
                for key, value in self.results.items():
                    txt_file.write(f"{key}: {value}\n")
            print(f"Results saved to {file_path}.txt")

        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def to_nir(self, dummy_input: Tensor, filename: str, **kwargs) -> None:
        """
        Exports the model to the NIR (Neural Intermediate Representation) format.

        Args:
            dummy_input (torch.Tensor):
                A sample input tensor that matches the input shape of the model.
                This is required for tracing the model during export.
            filename (str):
                The file path where the exported NIR file will be saved.
            **kwargs:
                Additional keyword arguments passed to the `export_to_nir` function
                for customization during the export process.

        Raises:
            ValueError:
                If the installed version of `snntorch` is less than `0.9.0`.

        """
        try:
            import nir
        except ImportError:
            raise ImportError(
                "Exporting to NIR requires the `nir` package. Install it using `pip install nir`."
            )
        if snntorch.__version__ < "0.9.0":
            raise ValueError("Exporting to NIR requires snntorch version >= 0.9.0")

        if snntorch.__version__ >= "0.9.0":
            from snntorch.export_nir import export_to_nir

        nir_graph = export_to_nir(self.model.__net__(), dummy_input, **kwargs)
        nir.write(filename, nir_graph)
        print(f"Model exported to {filename}")

    def to_onnx(self, dummy_input: Tensor, filename: str, **kwargs) -> None:
        """
        Exports the model to the ONNX (Open Neural Network Exchange) format.

        Args:
            dummy_input (torch.Tensor):
                A sample input tensor that matches the input shape of the model.
                This tensor is required for tracing the model during the export process.
            filename (str):
                The file path where the ONNX model will be saved, including the `.onnx` extension.
            **kwargs:
                Additional keyword arguments passed to the `torch.onnx.export` function
                for customization during the export process.

        Raises:
            RuntimeError:
                If an error occurs during the ONNX export process.

        """
        if dummy_input.requires_grad:
            dummy_input = dummy_input.detach()

        for param in self.model.__net__().parameters():
            param.requires_grad = False

        if isinstance(self.model, SNNTorchModel):
            utils.reset(self.model.__net__())

        for buffer in self.model.__net__().buffers():
            buffer.requires_grad = False

        with torch.no_grad():
            self.model.__net__().eval()
            torch.onnx.export(self.model.__net__(), dummy_input, filename, **kwargs)
        print(f"Model exported to {filename}")
