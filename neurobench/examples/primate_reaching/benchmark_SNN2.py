import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.benchmarks import Benchmark
from neurobench.examples.primate_reaching.SNN2 import SNN2

from neurobench.models import TorchModel
import snntorch as snn

# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

# The dataloader and preprocessor has been combined together into a single class
files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02",
            "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]

footprint = []
connection_sparsity = []
activation_sparsity = []
dense = []
macs = []
acs = []
r2 = []


for filename in files:
    print("Processing {}".format(filename))
    dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename=filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, remove_segments_inactive=False)
    
    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)

    net = SNN2(input_size=dataset.input_feature_size)
    net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/SNN2_{}.pt".format(filename), map_location=torch.device('cpu'))
                        ['model_state_dict'], strict=False)

    # init the model
    net.reset()
    model = TorchModel(net) # using TorchModel instead of SNNTorchModel because the SNN iterates over dimension 0
    model.add_activation_module(snn.SpikingNeuron)

    static_metrics = ["model_size", "connection_sparsity"]
    workload_metrics = ["r2", "activation_sparsity", "synaptic_operations"]

    # Benchmark expects the following:
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, workload_metrics])
    results = benchmark.run()
    print(results)

    footprint.append(results['model_size'])
    connection_sparsity.append(results['connection_sparsity'])
    activation_sparsity.append(results['activation_sparsity'])
    dense.append(results['synaptic_operations']['Dense'])
    macs.append(results['synaptic_operations']['Effective_MACs'])
    acs.append(results['synaptic_operations']['Effective_ACs'])
    r2.append(results['r2'])

print("Footprint: {}".format(footprint))
print("Connection sparsity: {}".format(connection_sparsity))
print("Activation sparsity: {}".format(activation_sparsity), sum(activation_sparsity)/len(activation_sparsity))
print("Dense: {}".format(dense), sum(dense)/len(dense))
print("MACs: {}".format(macs), sum(macs)/len(macs))
print("ACs: {}".format(acs), sum(acs)/len(acs))
print("R2: {}".format(r2), sum(r2)/len(r2))

# Footprint: [19648, 19648, 19648, 38848, 38848, 38848]
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Activation sparsity: [0.9963387091440609, 0.9968660155513294, 0.9963804923837362, 0.9987306432212195, 0.9986144471669343, 0.9988117468476962] 0.9976236757191628
# Dense: [4900.0, 4900.0, 4900.0, 9700.0, 9700.0, 9700.0] 7300.0
# MACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# ACs: [396.6414365765915, 244.43293219292173, 185.92106889094842, 510.3726604741439, 558.8291751660652, 584.9244337613446] 413.5202845103359
# R2: [0.6774135828018188, 0.5010538101196289, 0.5994032025337219, 0.5707334280014038, 0.5145009756088257, 0.6201670169830322] 0.5805453360080719