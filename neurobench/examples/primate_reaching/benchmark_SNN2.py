import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
# from neurobench.models.snntorch_models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.examples.primate_reaching.SNN2 import SNN2

from neurobench.models import TorchModel
import snntorch as snn

# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02",
            "loco_20170210_03", "loco_20170217_02", "loco_20170301_05"]

footprint = []
connection_sparsity = []
activation_sparsity = []
macs = []
acs = []
r2 = []


for filename in files:
    print("Processing {}".format(filename))
    dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename=filename,
                              num_steps=1, train_ratio=0.5, bin_width=0.004,
                              biological_delay=0, split_num=1, remove_segments_inactive=True)
    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)

    net = SNN2(input_size=dataset.input_feature_size)
    net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/SNN2_{}.pt".format(filename), map_location=torch.device('cpu'))
                        ['model_state_dict'], strict=False)

    # init the model
    net.reset()
    model = TorchModel(net) # using TorchModel instead of SNNTorchModel because the SNN iterates over dimension 0
    model.add_activation_module(snn.SpikingNeuron)

    static_metrics = ["model_size", "connection_sparsity"]
    data_metrics = ["r2", "activation_sparsity", "synaptic_operations"]

    # Benchmark expects the following:
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics])
    results = benchmark.run()
    print(results)

    footprint.append(results['model_size'])
    connection_sparsity.append(results['connection_sparsity'])
    activation_sparsity.append(results['activation_sparsity'])
    macs.append(results['synaptic_operations']['MACs'])
    acs.append(results['synaptic_operations']['ACs'])
    r2.append(results['r2'])

print("Footprint: {}".format(footprint))
print("Connection sparsity: {}".format(connection_sparsity))
print("Activation sparsity: {}".format(activation_sparsity), sum(activation_sparsity)/len(activation_sparsity))
print("MACs: {}".format(macs), sum(macs)/len(macs))
print("ACs: {}".format(acs), sum(acs)/len(acs))
print("R2: {}".format(r2), sum(r2)/len(r2))


# Footprint: [19648, 19648, 19648, 38848, 38848, 38848]
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Activation sparsity: [0.9965175875485318, 0.997256243446036, 0.9963114291485469, 0.9982373254459739, 0.9984370799487636, 0.9981614586393999] 0.997486854029542
# MACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# ACs: [397.61693326468657, 251.84787677020657, 185.9262434352796, 545.7270819308598, 608.6871787843487, 646.6091800356506] 439.4024157035053
# R2: [0.6782432794570923, 0.4903080463409424, 0.6044600009918213, 0.5680279731750488, 0.5561697483062744, 0.6059783101081848] 0.583864559729894