import torch
from torch.utils.data import DataLoader, Subset
import snntorch as snn
from snntorch import surrogate

from neurobench.datasets import PrimateReaching
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.primate_reaching.SNN_3 import SNNModel3

# Download data to /data/primate_reaching/PrimateReachingDataset. See PrimateReaching
# class for download instructions.

all_files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02", 
             "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]

footprint = []
connection_sparsity = []
activation_sparsity = []
dense = []
macs = []
acs = []
r2 = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
for filename in all_files:
    print("Processing {}".format(filename))

    # The dataloader and preprocessor has been combined together into a single class
    dataset = PrimateReaching(file_path="data/primate_reaching/PrimateReachingDataset/", filename=filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, remove_segments_inactive=False)
    
    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=1024, shuffle=False)

    net = SNNModel3(input_dim=dataset.input_feature_size, layer1=32, layer2=48, output_dim=2,
                    batch_size=256, bin_window=0.2, num_steps=7, drop_rate=0.5,
                    beta=0.5, mem_thresh=0.5, spike_grad=surrogate.atan(alpha=2))

    # Give the user the option to load their pretrained weights
    # TODO: currently model is not trained
    net.load_state_dict(torch.load("neurobench/examples/primate_reaching/model_data/SNN3_Weight/"+filename+"_model_state_dict.pth", map_location=device))

    model = TorchModel(net)
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
