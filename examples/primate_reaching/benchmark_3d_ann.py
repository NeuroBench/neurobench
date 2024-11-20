import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark

from ANN import ANNModel3D

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
    data_dir = "../../../data/primate_reaching/PrimateReachingDataset/" # data in repo root dir
    dataset = PrimateReaching(file_path=data_dir, filename=filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, remove_segments_inactive=False)

    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=256, shuffle=False)

    net = ANNModel3D(input_dim=dataset.input_feature_size, layer1=32, layer2=48, 
                     output_dim=2, bin_window=0.2, num_steps=7, drop_rate=0.5)

    net.load_state_dict(torch.load("./model_data/3D_ANN_Weight/"+filename+"_model_state_dict.pth", map_location=device))

    model = TorchModel(net)

    static_metrics = ["footprint", "connection_sparsity"]
    workload_metrics = ["r2", "activation_sparsity", "synaptic_operations"]

    # Benchmark expects the following:
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, workload_metrics])
    results = benchmark.run(device=device)
    print(results)

    footprint.append(results['footprint'])
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

# Footprint: [94552, 94552, 94552, 180952, 180952, 180952]
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Activation sparsity: [0.7647354121258293, 0.7170055772865946, 0.5856741871331477, 0.732334983618757, 0.5229688072074897, 0.7613369241509738] 0.6806759819204653
# Dense: [23136.0, 23136.0, 23136.0, 44640.0, 44640.0, 44640.0] 33888.0
# MACs: [10443.117328969101, 7005.948456888324, 6213.918443002781, 14398.115916437426, 14647.870704534756, 16333.473996765613] 11507.074141099665
# ACs: [0.006274841054957653, 0.002096161404428141, 0.005560704355885079, 0.011674826884207607, 0.010879230085255504, 0.007352486324552179] 0.007306375018214362
# R2: [0.6739100217819214, 0.5933212041854858, 0.6587119698524475, 0.5970449447631836, 0.5281975269317627, 0.6410964727401733] 0.6153803567091624
