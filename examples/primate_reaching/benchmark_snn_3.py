import torch
from torch.utils.data import DataLoader, Subset
import snntorch as snn
from snntorch import surrogate

from neurobench.datasets import PrimateReaching
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from SNN_3 import SNNModel3

all_files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02", 
             "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]

footprint = []
connection_sparsity = []
activation_sparsity = []
dense = []
macs = []
acs = []
r2 = []

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # workload runs faster on CPU due to I/O
    
for filename in all_files:
    print("Processing {}".format(filename))

    # The dataloader and preprocessor has been combined together into a single class
    data_dir = "../../../data/primate_reaching/PrimateReachingDataset/" # data in repo root dir
    dataset = PrimateReaching(file_path=data_dir, filename=filename,
                            num_steps=1, train_ratio=0.5, bin_width=0.004,
                            biological_delay=0, remove_segments_inactive=False)
    
    test_set_loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=1024, shuffle=False)

    net = SNNModel3(input_dim=dataset.input_feature_size, layer1=32, layer2=48, output_dim=2,
                    batch_size=256, bin_window=0.2, num_steps=7, drop_rate=0.5,
                    beta=0.5, mem_thresh=0.5, spike_grad=surrogate.atan(alpha=2))

    net.load_state_dict(torch.load("./model_data/SNN3_Weight/"+filename+"_model_state_dict.pth", map_location=device))

    model = TorchModel(net)
    model.add_activation_module(snn.SpikingNeuron)

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

# Footprint: [24972, 24972, 24972, 43020, 43020, 43020]
# Connection sparsity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Activation sparsity: [0.6453041225691365, 0.6873141238678099, 0.919185901113612, 0.9157824548894764, 0.7036032468324952, 0.8569459155448776] 0.788022627469568
# Dense: [32928.0, 32928.0, 32928.0, 54432.0, 54432.0, 54432.0] 43680.0
# MACs: [21504.0, 21503.999401096742, 21504.0, 43007.99836552424, 43007.99707097652, 43007.99943442413] 32255.999045336936
# ACs: [6321.460861433107, 7215.396642398608, 5009.047845227062, 4860.221515246594, 5612.495449552801, 5969.860638570507] 5831.413825404779
# R2: [0.6967655420303345, 0.5771909952163696, 0.6517471075057983, 0.6225693821907043, 0.56768798828125, 0.681292712688446] 0.6328756213188171
