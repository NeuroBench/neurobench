# ######################################################################################## ##
# Human Activity
# Recognition (HAR) is a time-dependent task that has applications in various aspects of human life, from healthcare
# to sports, safety, and smart environments. In this task, we present a comparative analysis of different SNN-based
# models designed for classifying raw signals (Accelerometer and Gyroscope) collected in the Wireless Sensor Data
# Mining (WISDM) dataset. ###The WISDM dataset consists of data from 51 subjects performing 18 activities. This
# dataset collects signals from both the accelerometer and the gyroscope of a smartphone and a smartwatch. Each
# activity is recorded for 3 minutes with an acquisition rate of 20 Hz. The dataset's classes are balanced,
# with each activity represented in the dataset contributing approximately 5.3% to 5.8% of the total approximately
# 15.63 million samples. ###From the whole smartwatch dataset, we selected a subset of general hand-oriented
# activities for our analysis. These activities include: (1) dribbling in basketball, (2) playing catch with a tennis
# ball, (3) typing, (4) writing, (5) clapping, (6) brushing teeth, and (7) folding clothes. We divided the signals
# into non-overlapping temporal windows with a length of 2 seconds. These temporal windows serve as the input layer
# for the benchmarked models.
# ########################################################################################
from neurobench.datasets.WISDM_data_loader import WISDMDataModule
from training import SpikingNetwork
from neurobench.accumulators.accumulator import choose_max_count
from neurobench.benchmarks import Benchmark
from neurobench.models import SNNTorchModel


if __name__ == '__main__':
    batch_size = 256
    lr = 1.e-3
    dataset_path = "./dataset/watch_subset2_40.npz"
    data_module = WISDMDataModule(dataset_path, batch_size=batch_size)
    data_module.setup('test')

    num_inputs = data_module.num_inputs
    num_outputs = data_module.num_outputs
    num_steps = data_module.num_steps

    spiking_network = SpikingNetwork.load_from_checkpoint('./model_data/WISDM_snnTorch.ckpt', map_location='cpu')

    model = SNNTorchModel(spiking_network.model)
    test_set_loader = data_module.test_dataloader()

    # # # postprocessors
    postprocessors = [choose_max_count]
    # #
    static_metrics = ["model_size"]
    data_metrics = ["classification_accuracy"]
    # #
    benchmark = Benchmark(model, test_set_loader, [], postprocessors, [static_metrics, data_metrics])
    results = benchmark.run()
    print(results)