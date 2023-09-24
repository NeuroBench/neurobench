import torch
from torch import nn

from torch.utils.data import Subset, DataLoader

import wandb
import argparse
import pandas as pd

from neurobench.datasets import MackeyGlass
from neurobench.models import TorchModel
from neurobench.benchmarks import Benchmark

from neurobench.examples.mackey_glass.echo_state_network import EchoStateNetwork
from neurobench.examples.mackey_glass.lstm_model import LSTMModel
torch.autograd.set_detect_anomaly(True)

mg_parameters_file="neurobench/datasets/mackey_glass_parameters.csv"
mg_parameters = pd.read_csv(mg_parameters_file)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--wb', dest='wandb_state', type=str, default="offline", help="wandb state")
parser.add_argument('--name', type=str, default='LSTM_MG', help='wandb run name')
parser.add_argument('--project', type=str, default='Neurobench', help='wandb project name')
parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--series_id', type=int, default=0)
parser.add_argument('--repeat', type=int, default=1)
# seed set by the repeat id
#parser.add_argument('--seed', type=int, default=41)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--sw', type=bool, default=False, help="activate wb sweep run")
parser.add_argument('--debug', type=bool, default=False)

args, unparsed = parser.parse_known_args()

if args.sw:
    wandb.init(name=args.name,
               mode=args.wandb_state,
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
start_offset_range = torch.arange(0., 0.5*args.repeat, 0.5) 

# benchmark run over args.repeat different experiments
sMAPE_scores = []

for repeat_id in range(args.repeat):

    # set seed for RG for reproducible results
    torch.manual_seed(repeat_id)
    torch.cuda.manual_seed_all(repeat_id)

    mg = MackeyGlass(tau = mg_parameters.tau[args.series_id], 
                     lyaptime = mg_parameters.lyapunov_time[args.series_id],
                     constant_past = mg_parameters.initial_condition[args.series_id],
                     start_offset=start_offset_range[repeat_id].item(),
                     bin_window=1)

    train_set = Subset(mg, mg.ind_train)
    test_set = Subset(mg, mg.ind_test)

    # Initialize an LSTM model
    lstm = LSTMModel(**params)
    lstm.to(device)

    # LSTM training phase
    lstm.train()

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(lstm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_data, train_labels = train_set[:]

    warmup = 0.6 # in Lyapunov times
    warmup_pts = round(warmup*mg.pts_per_lyaptime)
    train_labels = train_labels[warmup_pts:]

    # training loop
    for epoch in range(args.n_epochs):

        train_data = train_data.to(device)
        train_labels = train_labels.to(device)

        pre = lstm(train_data)

        loss_val = criterion(pre[warmup_pts:,:],
                                 train_labels)

        opt.zero_grad()
        loss_val.backward()
        opt.step()

        print(f"Epoch {epoch}: loss = {loss_val.item()}")
        wandb.log({"loss": loss_val.item()})

    if args.debug:
        import matplotlib.pyplot as plt        

        fig = plt.figure()

        plt.plot(pre[:,0].detach().cpu().numpy(), label='prediction')
        plt.plot(train_labels[:,0].detach().cpu().numpy(), label='target')

        plt.xlabel('time')
        plt.legend()

        print("saved training fit to ./fit_train.pdf")
        wandb.log({f'fig_train': wandb.Image(fig)})  
        #plt.savefig("fit_train.pdf")
        plt.close()
 
    ## Testing ##
    test_set_loader = DataLoader(train_set, batch_size=mg.testtime_pts, shuffle=False)
    lstm.mode = "autonomous"
    lstm.device = torch.device("cpu")
    lstm.to(torch.device("cpu"))
    model = TorchModel(lstm)

    static_metrics = ["model_size", "connection_sparsity"]
    data_metrics = ["sMAPE", "activation_sparsity"]

    benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics]) 
    results = benchmark.run()
    print(results)
    sMAPE_scores.append(results["sMAPE"])

connection_sparsity = results["connection_sparsity"]
model_size = results["model_size"]

avg_sMAPE_score = sum(sMAPE_scores)/len(sMAPE_scores)
wandb.log({"sMAPE_score": avg_sMAPE_score})
wandb.log({"connection_sparsity": connection_sparsity})
wandb.log({"model_size": model_size})

print(f"sMAPE score {avg_sMAPE_score} on time series id {args.series_id}")
