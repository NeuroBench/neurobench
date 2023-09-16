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

mg_parameters_file="neurobench/datasets/mackey_glass_parameters.csv"
mg_parameters = pd.read_csv(mg_parameters_file)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--wb', dest='wandb_state', type=str, default="offline", help="wandb state")
parser.add_argument('--name', type=str, default='LSTM_MG', help='wandb run name')
parser.add_argument('--project', type=str, default='Neurobench', help='wandb project name')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--series_id', type=int, default=0)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=41)
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

# set seed for RG for reproducible results
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\nCUDA enabled")
else:
    device = torch.device("cpu")
    print("\nCUDA not available")

# LSTM parameters
params = {}
params['input_dim'] = 1
params['hidden_size'] = args.hidden_size
params['n_layers'] = args.n_layers
params['output_dim'] = 1
params['dropout_rate'] = args.dropout_rate
params['dtype'] = torch.float64
params['mode'] = 'single_step'

# benchmark run over 14 different series
sMAPE_scores = []

mg = MackeyGlass(tau = mg_parameters.tau[args.series_id], 
                 lyaptime = mg_parameters.lyapunov_time[args.series_id],
                 constant_past = mg_parameters.initial_condition[args.series_id],
                 start_offset=0.)

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

train_data = train_data.permute(1,0,2) # (batch, timesteps, features)
warmup = 0.6 # in Lyapunov times
warmup_pts = round(warmup*mg.pts_per_lyaptime)
train_labels = train_labels[warmup_pts:]

# training loop
for epoch in range(args.n_epochs):

    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    #break
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

    plt.figure()

    plt.plot(pre[:,0].detach().cpu().numpy(), label='prediction')
    plt.plot(train_labels[:,0].detach().cpu().numpy(), label='target')

    plt.xlabel('time')
    plt.legend()

    print("saved training fit to ./fit_train.pdf")
    plt.savefig("fit_train.pdf")

#torch.save(lstm, 'neurobench/examples/mackey_glass/model_data/lstm.pth')
 
## Load Model ##
#net = torch.load('neurobench/examples/mackey_glass/model_data/lstm.pth')
test_set_loader = DataLoader(train_set, batch_size=mg.testtime_pts, shuffle=False)
lstm.mode = 'autonomous'
lstm.to(torch.device("cpu"))
#net.mode = 'autonomous'
model = TorchModel(lstm)
#model = TorchModel(net)
# data_metrics = ["activation_sparsity", "multiply_accumulates", "sMAPE"]

static_metrics = ["model_size", "connection_sparsity"]
data_metrics = ["sMAPE"]

benchmark = Benchmark(model, test_set_loader, [], [], [static_metrics, data_metrics]) 
results = benchmark.run()
print(results)
sMAPE_score = results["sMAPE"]
connection_sparsity = results["connection_sparsity"]
model_size = results["model_size"]

wandb.log({"sMAPE_score": sMAPE_score})
wandb.log({"connection_sparsity": connection_sparsity})
wandb.log({"model_size": model_size})

print(f"sMAPE score {sMAPE_score} on time series id {args.series_id}")

