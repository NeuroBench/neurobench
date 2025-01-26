import os
import torch
from torch.utils.data import DataLoader, Subset

from neurobench.datasets import PrimateReaching
from neurobench.models.torch_model import TorchModel
from neurobench.benchmarks import Benchmark

from ANN import ANNModel2D

from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    R2
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)

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

file_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_path, "../../data/primate_reaching/PrimateReachingDataset/") # data in repo root dir

for filename in all_files:
    print("Processing {}".format(filename))

    # The dataloader and preprocessor has been combined together into a single class
    dataset = PrimateReaching(file_path=data_dir, filename=filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, remove_segments_inactive=False)

    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=256, shuffle=False)

    net = ANNModel2D(input_dim=dataset.input_feature_size, layer1=32, layer2=48, 
                     output_dim=2, bin_window=0.2, drop_rate=0.5)

    model_path = os.path.join(file_path, "model_data/2D_ANN_Weight/"+filename+"_model_state_dict.pth")
    net.load_state_dict(torch.load(model_path, map_location=device))

    model = TorchModel(net)

    static_metrics = [Footprint, ConnectionSparsity]
    workload_metrics = [R2, ActivationSparsity, SynapticOperations]

    # Benchmark expects the following:
    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, workload_metrics])
    results = benchmark.run(device=device)
    print(results)

    footprint.append(results['Footprint'])
    connection_sparsity.append(results['ConnectionSparsity'])
    activation_sparsity.append(results['ActivationSparsity'])
    dense.append(results['SynapticOperations']['Dense'])
    macs.append(results['SynapticOperations']['Effective_MACs'])
    acs.append(results['SynapticOperations']['Effective_ACs'])
    r2.append(results['R2'])

print("Footprint: {}".format(footprint))
print("Connection sparsity: {}".format(connection_sparsity))
print("Activation sparsity: {}".format(activation_sparsity), sum(activation_sparsity)/len(activation_sparsity))
print("Dense: {}".format(dense), sum(dense)/len(dense))
print("MACs: {}".format(macs), sum(macs)/len(macs))
print("ACs: {}".format(acs), sum(acs)/len(acs))
print("R2: {}".format(r2), sum(r2)/len(r2))

# Footprint: [20824, 20824, 20824, 33496, 33496, 33496]
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Activation sparsity: [0.7068512007122443, 0.7274494314849341, 0.6142621034584272, 0.6290474755671983, 0.6793054885963405, 0.6963649652600741] 0.6755467775132032
# Dense: [4702.261627687736, 4701.8430499148435, 4699.549582947173, 7773.2197567257945, 7771.01773105288, 7772.632844051291] 6236.754098729952
# MACs: [4306.322415210456, 3595.209672287623, 3607.261044176707, 5851.9819915795315, 5995.014802029395, 6462.786839756449] 4969.76279417336
# ACs: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.0
# R2: [0.6327020525932312, 0.5241347551345825, 0.6216747164726257, 0.5727078914642334, 0.4745999276638031, 0.6272222995758057] 0.5755069404840469