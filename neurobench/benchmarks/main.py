import sys
import argparse
sys.path.append("../..")
from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.models import ANNModel, SNNModel, LSTMModel
from neurobench.benchmarks.benchmark import Benchmark
import yaml
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="ANN")
    args = parser.parse_args()

    with open('hyperparams.yaml') as f:
        hyperparams = yaml.load(f, Loader=yaml.loader.SafeLoader)

    if torch.cuda.is_available():
        print("using cuda")
        hyperparams['device'] = torch.device("cuda")
    else:
        print("using cpu")
        hyperparams['device'] = torch.device("cpu")

    if args.model_name == 'LSTM':
        # num_steps is required 3D mode
        ds = PrimateReaching(path=hyperparams['dataset_file'], filename=hyperparams['filename'],
                            postpr_data_path=hyperparams['postpr_data_path'], regenerate=False,
                            biological_delay=0, spike_sorting=False, Np=25,
                            mode="3D", advance=0.016, bin_width=0.08, num_steps=hyperparams['num_steps'], batch_size=hyperparams['batch_size'])
    else:    
        ds = PrimateReaching(path=hyperparams['dataset_file'], filename=hyperparams['filename'],
                            postpr_data_path=hyperparams['postpr_data_path'], regenerate=False,
                            biological_delay=0, spike_sorting=False, Np=25,
                            mode="2D", advance=0.016, bin_width=0.08)
    if args.model_name == 'LSTM':
        net = LSTMModel()
    elif args.model_name == 'ANN':
        net = ANNModel(input_dim=96, layer1=32, layer2=48, output_dim=2, dropout_rate=0.5)
    elif args.model_name == 'SNN':
        net = SNNModel(beta=0.5, mem_threshold=0.3, input_dim=96, layer1=32, layer2=48,
                       output_dim=2, dropout_rate=0.5, num_step=15, data_mode="2D")
    else:
        print(f"Model {model_name} not implemented yet")
        exit()

    benchmark = Benchmark(dataset=ds, net=net, hyperparams=hyperparams, model_type=args.model_name)

    benchmark.run()
