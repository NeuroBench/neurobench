import argparse
import pandas as pd
try:
    import wandb
    loaded_wandb = True
except:
    loaded_wandb = False

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.mackey_glass.lstm_model import LSTMModel

mg_parameters_file = "neurobench/datasets/mackey_glass_parameters.csv"
mg_parameters = pd.read_csv(mg_parameters_file)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--wb', dest='wandb_state', type=str, default="offline")
parser.add_argument('--name', type=str, default='LSTM_MG')
parser.add_argument('--project', type=str, default='Neurobench')
parser.add_argument('--input_dim', type=int, default=50)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=40)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--series_id', type=int, default=0)
parser.add_argument('--repeat', type=int, default=30)
# seed set by the repeat id
# parser.add_argument('--seed', type=int, default=41)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--sw', type=bool, default=False)
parser.add_argument('--debug', type=bool, default=False)

args, unparsed = parser.parse_known_args()

assert args.series_id == 0, "Hyperparameter optimization performed "\
                            "only for series id 0"

if loaded_wandb:
    if args.sw:
        wandb.init(mode=args.wandb_state,
                   config=wandb.config)

        config_wb = wandb.config
    else:
        wandb.init(project=args.project,
                   name=args.name,
                   mode=args.wandb_state)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nCUDA enabled")
else:
    device = torch.device("cpu")
    print("\nCUDA not available")

# LSTM parameters
params = {}
params['input_dim'] = args.input_dim
params['hidden_size'] = args.hidden_size
params['n_layers'] = args.n_layers
params['output_dim'] = 1
params['dtype'] = torch.float64
params['device'] = device
params['mode'] = 'single_step'

# Benchmark run over args.repeat different experiments
sMAPE_scores = []
connection_sparsities = []
activation_sparsities = []
synop_macs = []
synop_dense = []

# Shift time series by 0.5 of Lyapunov time-points for each independent run 
start_offset_range = torch.arange(0., 0.5*args.repeat, 0.5) 
lyaptime_pts = 75
start_offset_range = start_offset_range * lyaptime_pts

data_dir = "data/mackey_glass/"

for repeat_id in range(args.repeat):
    tau = mg_parameters.tau[args.series_id]
    filepath = data_dir + "mg_" + str(tau) + ".npy"
    lyaptime = mg_parameters.lyapunov_time[args.series_id]
    constant_past = mg_parameters.initial_condition[args.series_id]
    offset = start_offset_range[repeat_id].item()

    print(f"Experiment: repeat-id={repeat_id},\n"
          f"tau={tau}, constant_past={constant_past},\n"
          f"lyaptime={lyaptime}, offset={offset}")

    if repeat_id == 0 and loaded_wandb:
        wandb.config['tau'] = tau
        wandb.config['lyaptime'] = lyaptime
        wandb.config['constant_past'] = constant_past
        wandb.config['offset'] = offset
        wandb.config['repeat'] = args.repeat

    # set seed for RG for reproducible results
    torch.manual_seed(repeat_id)
    torch.cuda.manual_seed_all(repeat_id)

    mg = MackeyGlass(filepath,
                     start_offset=offset,
                     bin_window=1)

    train_set = Subset(mg, mg.ind_train)
    test_set = Subset(mg, mg.ind_test)

    # Initialize an LSTM model
    lstm = LSTMModel(**params)
    lstm.to(device)

    # LSTM training phase
    lstm.train()

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(lstm.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    train_data, train_labels = train_set[:]

    # Training loop
    for epoch in range(args.n_epochs):

        train_data = train_data.to(device)
        train_labels = train_labels.to(device)

        pre = lstm(train_data)

        loss_val = criterion(pre,
                             train_labels)

        opt.zero_grad()
        loss_val.backward()
        opt.step()

        print(f"Epoch {epoch}: loss = {loss_val.item()}")
        if loaded_wandb:
            wandb.log({"loss": loss_val.item()})

    if args.debug and loaded_wandb:
        import matplotlib.pyplot as plt

        fig = plt.figure()

        plt.plot(pre[:, 0].detach().cpu().numpy(), label='prediction')
        plt.plot(train_labels[:, 0].detach().cpu().numpy(), label='target')

        plt.xlabel('time')
        plt.legend()

        print("loaded training fit to wandb")
        wandb.log({'fig_train': wandb.Image(fig)})
        plt.close()

    # Testing
    test_set_loader = DataLoader(train_set,
                                 batch_size=mg.testtime_pts,
                                 shuffle=False)
    lstm.mode = "autonomous"
    lstm.device = torch.device("cpu")
    lstm.to(torch.device("cpu"))
    model = TorchModel(lstm)

    static_metrics = ["model_size", "connection_sparsity"]
    workload_metrics = ["sMAPE", "activation_sparsity", "synaptic_operations"]

    benchmark = Benchmark(model, test_set_loader, [], [],
                          [static_metrics, workload_metrics])
    results = benchmark.run()
    print(results)
    sMAPE_scores.append(results["sMAPE"])
    connection_sparsities.append(results["connection_sparsity"])
    activation_sparsities.append(results["activation_sparsity"])
    synop_macs.append(results["synaptic_operations"]["Effective_MACs"])
    synop_dense.append(results["synaptic_operations"]["Dense"])
    if loaded_wandb:
        wandb.log({"sMAPE_score_val": results["sMAPE"]})

model_size = results["model_size"]

avg_sMAPE_score = sum(sMAPE_scores)/args.repeat
connection_sparsity = sum(connection_sparsities)/args.repeat
activation_sparsity = sum(activation_sparsities)/args.repeat
synop_macs = sum(synop_macs)/args.repeat
synop_dense = sum(synop_dense)/args.repeat

if loaded_wandb:
    wandb.log({"sMAPE_score": avg_sMAPE_score})
    wandb.log({"connection_sparsity": connection_sparsity})
    wandb.log({"activation_sparsity": activation_sparsity})
    wandb.log({"model_size": model_size})

print(f"sMAPE score = {avg_sMAPE_score},\n"
      f"connection_sparsity = {connection_sparsity},\n"
      f"activation_sparsity = {activation_sparsity},\n"
      f"synop_macs = {synop_macs},\n"
      f"synop_dense = {synop_dense},\n"
      f"on time series id {args.series_id}")

# With the default params, repeat 30, tau=17
# sMAPE score = 15.156239883579927,
# connection_sparsity = 0.0,
# activation_sparsity = 0.45951777777777786,
# synop_macs = 14534.032622222225,
# synop_dense = 14552.413333333332,
# on time series id 0
