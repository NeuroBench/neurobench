import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate

from neurobench.datasets import DVSGesture
from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark
from neurobench.postprocessing.postprocessor import aggregate,choose_max_count

from CSNN import Conv_SNN

#wandb tracking
import wandb

run = wandb.init(project='dvs_gesture', 
        #    entity='neurobench',
           config={
               'model': 'CSNN',
               'dataset': 'dvs_gesture',
               'batch_size': 1,
               'learning_rate': 1.2e-3,
               'warmup_frames': 34,
               'nr_episodes': 10,
               'optimizer': 'Adamax',
               'betas': [0.9, 0.95]
           })

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# Set the seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# data in repo root dir
data = DVSGesture('../../../data/dvs_gesture/', split='testing', preprocessing='stack')
dataloader_training = DataLoader(data, 24,shuffle=False)
model = Conv_SNN().to(device)
run.watch(model, log='all', log_freq=1)
# data_1 = [(torch.tensor(data[0][0]).unsqueeze(0),torch.tensor(data[0][1]).unsqueeze(0))]
# data_2 = [next(iter(dataloader_training))]

# single_batch = next(iter(dataloader_training))
# print(single_batch[0].shape)
# torch.save(model.state_dict(), 'model_data/DVS_SNN_untrained.pth')

optimizer = torch.optim.Adamax(model.parameters(),lr=1.2e-3,betas=[0.9,0.95])
# model.fit(dataloader_training=dataloader_training,device=device, warmup_frames=70, optimizer=optimizer, nr_episodes=1000)
model.fit(dataloader_training=dataloader_training,device=device, warmup_frames=34, optimizer=optimizer, nr_episodes=3)
# torch.save(model.state_dict(), 'neurobench/examples/dvs_gesture/model_data/DVS_SNN_trained.pth')