import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate
from snntorch import utils
# datasets
from neurobench.datasets.dvs_gesture import DVSGesture
from neurobench.models import SNNTorchModel
from neurobench.postprocessing.postprocessor import choose_max_count

from tqdm import tqdm

import wandb
# from torch.profiler import profile, record_function, ProfilerActivity

class Conv_SNN(nn.Module):
	def __init__(self):
		super(Conv_SNN,self).__init__()
		beta = .9
		alpha = 0.95 # a 1st order if alpha = 0

		self.reduce = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
		grad = surrogate.fast_sigmoid()
		stride = 2
		self.pool1 = nn.AvgPool2d(2,stride=stride)
		self.conv1 = nn.Conv2d(3,24,3,1)
		# initialize the weights of the convolutional layer with a normal distribution with mean .5 and std 0.25
		nn.init.normal_(self.conv1.weight, mean=0.5, std=0.25)

		self.syn1  = snn.Leaky(beta= .5, spike_grad = grad)

		self.pool2 = nn.AvgPool2d(2,stride=stride)
		self.conv2 = nn.Conv2d(24,24,3, 1)
		nn.init.normal_(self.conv2.weight, mean=0.5, std=0.25)
		self.syn2  = snn.Leaky(beta= .7, spike_grad = grad)

		self.pool3 = nn.AvgPool2d(2,stride=stride)
		self.conv3 = nn.Conv2d(24, 64,3,1)
		nn.init.normal_(self.conv3.weight, mean=0.5, std=0.25)
		self.syn3  = snn.Leaky(beta= .8, spike_grad = grad)

		self.lin1  = nn.Linear(1024,11)
		self.syn4  = snn.Leaky(beta= .95, spike_grad = grad)

		self.lin2  = nn.Linear(128,11)

		self.mem1 = self.syn1.init_leaky()
		self.mem2 = self.syn2.init_leaky()
		self.mem3 = self.syn3.init_leaky()
		self.mem4 = self.syn4.init_leaky()


	def forward(self, frame, warmup_frames = 0):
		frame = self.reduce(frame).to(dtype=torch.float32)
		# frame = frame.to(dtype=torch.float32)
		# frame = transforms.Resize((32,32))(frame).to(dtype=torch.float32)
		x = self.conv1(frame)
		x = self.pool1(x)
		# print(x.shape)
		x, self.mem1 = self.syn1(x, self.mem1)
		
		x = self.conv2(x)
		x = self.pool2(x)
		x, self.mem2 = self.syn2(x, self.mem2)

		x = self.conv3(x)
		
		x, self.mem3 = self.syn3(x, self.mem3)

		x = x.view(x.shape[0],-1)
		x = self.lin1(x)
		
		spks, self.mem4 = self.syn4(x, self.mem4)
		return spks.reshape(-1,11).detach(), self.mem4
	def reset(self):
		self.mem1 = self.syn1.init_leaky()
		self.mem2 = self.syn2.init_leaky()
		self.mem3 = self.syn3.init_leaky()
		self.mem4 = self.syn4.init_leaky()

	def single_forward(self, frames, warmup_frames = 0, wandb_log = False):
		self.reset()

		out_spk = 0
		# from [nr_batches,nr_frames,c,h,w] -> [nr_frames,nr_batches,c,h,w]
		# frames = frames.transpose(1,0)
		spk_1 = []
		spk_2 = []
		spk_3 = []
		spk_4 = []

		# Data is expected to be shape (batch, timestep, features*)
		for step in range(frames.shape[1]):
			frame = frames[:,step,:,:,:]
			frame = self.reduce(frame).to(dtype=torch.float32)
			# frame = frame.to(dtype=torch.float32)
			# frame = transforms.Resize((32,32))(frame).to(dtype=torch.float32)
			x = self.conv1(frame)
			x = self.pool1(x)
			x, self.mem1 = self.syn1(x, self.mem1)
			spk_1.append(x)
			if wandb_log:
				# wandb.log({'spikes1':x})
				# average number of spikes in layer 1
				spikes_avg = torch.mean(x)
				wandb.log({"avg membrane potential layer 1 in last training round" : torch.mean(self.mem1).item()})
				wandb.log({"avg nr of spikes layer 1 in last training round" : spikes_avg.item()})

				# Log y with WandB
				# wandb.log({f'spikes_1': [(step, y.item())]})
			x = self.conv2(x)
			x = self.pool2(x)
			x, self.mem2 = self.syn2(x, self.mem2)
			spk_2.append(x)
			if wandb_log:
				spikes_avg = torch.mean(x)
				wandb.log({"avg nr of spikes layer 2 in last training round" : spikes_avg.item()})
			x = self.conv3(x)
			
			x, self.mem3 = self.syn3(x, self.mem3)
			spk_3.append(x)
			if wandb_log:
				spikes_avg = torch.mean(x)
				wandb.log({"avg nr of spikes layer 3 in last training round" : spikes_avg.item()})
			x = x.view(x.shape[0],-1)
			x = self.lin1(x)
			
			x, self.mem4 = self.syn4(x, self.mem4)
			spk_4.append(x)
			if wandb_log:
				spikes_avg = torch.mean(x)
				wandb.log({"avg nr of spikes layer 4 in last training round" : spikes_avg.item()})

			if step >= warmup_frames:
				out_spk += x

		# log average spike rates of each layer
		spk_1 = torch.stack(spk_1)
		spk_2 = torch.stack(spk_2)
		spk_3 = torch.stack(spk_3)
		spk_4 = torch.stack(spk_4)

		wandb.log({'average spike rate of layer 1': torch.mean(spk_1)})
		wandb.log({'average spike rate of layer 1': torch.mean(spk_2)})
		wandb.log({'average spike rate of layer 3': torch.mean(spk_3)})
		wandb.log({'average spike rate of layer 4': torch.mean(spk_4)})
		# log a table of the output spikes with wandb
		
		# print(out_spk)
		prediction = torch.nn.functional.softmax(out_spk.reshape(11,-1), dim=0)
		return prediction
	
	def fit(self, dataloader_training, warmup_frames, optimizer, device, nr_episodes = 10):
		for episode in tqdm(range(nr_episodes)):
			# print(f"Episode {episode}")
			for frames, labels in dataloader_training:
				# Print GPU memory
				# with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
					# with record_function("model_inference"):
				wandb_log = False
				if episode == nr_episodes-1:
					wandb_log = True
				prediction = self.single_forward(frames.to(device), warmup_frames, wandb_log=wandb_log)

				# Add a delay to allow profiler to collect data
				# import time
				# time.sleep(5)
				# print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
				# prof.export_chrome_trace("trace.json")
				# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=10))

				label_tensor = labels.clone().detach().to(device)
				targets_one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=11).transpose(1,0)

				loss = torch.nn.functional.smooth_l1_loss(prediction,targets_one_hot)
				


				optimizer.zero_grad()
				loss.backward()
				wandb.log({'loss': loss.item()})
				optimizer.step()
			# print(loss.item())



