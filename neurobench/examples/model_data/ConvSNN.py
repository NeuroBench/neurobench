import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate

# datasets
from neurobench.datasets.DVSGesture_loader import DVSGesture
from neurobench.models import SNNTorchModel
from neurobench.accumulators.accumulator import choose_max_count

from tqdm import tqdm


class Conv_SNN(nn.Module):
	def __init__(self):
		super(Conv_SNN,self).__init__()
		beta = .9
		alpha = .95

		self.reduce = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
		grad = surrogate.fast_sigmoid()
		stride = 2
		self.pool1 = nn.AvgPool2d(2,stride=stride)
		self.conv1 = nn.Conv2d(3,32,5,1)
		self.syn1  = snn.Synaptic(alpha=alpha, beta= beta, spike_grad = grad)

		self.pool2 = nn.AvgPool2d(2,stride=stride)
		self.conv2 = nn.Conv2d(32,32,5,1)
		self.syn2  = snn.Synaptic(alpha=alpha, beta= beta, spike_grad = grad)

		self.pool3 = nn.AvgPool2d(2,stride=stride)
		self.conv3 = nn.Conv2d(32,11,5,1)
		self.syn3  = snn.Synaptic(alpha=alpha, beta= beta, spike_grad = grad)


		self.mem1, self.cur1 = self.syn1.init_synaptic()
		self.mem2, self.cur2 = self.syn2.init_synaptic()
		self.mem3, self.cur3 = self.syn3.init_synaptic()


	def forward(self, frame, warmup_frames = 0):
		out_spk = 0
		frame = self.reduce(frame).to(dtype=torch.float32)

		x = self.conv1(frame)
		x = self.pool1(x)
		x, self.mem1, self.cur1 = self.syn1(x, self.mem1,self.cur1)
		
		x = self.conv2(x)
		x = self.pool2(x)
		x, self.mem2, self.cur2 = self.syn2(x, self.mem2,self.cur2)

		x = self.conv3(x)
		out_spk, self.mem3, self.cur3 = self.syn3(x, self.mem3,self.cur3)

		prediction = out_spk.reshape(-1,11)
		
		return prediction, self.mem3
	
	def single_forward(self, frames, warmup_frames = 0):
		self.mem1, self.cur1 = self.syn1.init_synaptic()
		self.mem2, self.cur2 = self.syn2.init_synaptic()
		self.mem3, self.cur3 = self.syn3.init_synaptic()

		out_spk = 0
		# from [nr_batches,nr_frames,c,h,w] -> [nr_frames,nr_batches,c,h,w]
		frames = frames.transpose(1,0)
		for i, frame in enumerate(frames):
			frame = self.reduce(frame).to(dtype=torch.float32)
			# frame = transforms.Resize((32,32))(frame).to(dtype=torch.float32)
			x = self.conv1(frame)
			x = self.pool1(x)
			x, self.mem1, self.cur1 = self.syn1(x, self.mem1,self.cur1)
			
			x = self.conv2(x)
			x = self.pool2(x)
			x, self.mem2, self.cur2 = self.syn2(x, self.mem2,self.cur2)
			
			x = self.conv3(x)
			x, self.mem3, self.cur3 = self.syn3(x, self.mem3,self.cur3)
			if i >= warmup_frames:
				out_spk += x
		
		prediction = torch.nn.functional.softmax(out_spk.reshape(11,-1), dim=0)
		
		return prediction
	
	def fit(self, dataloader_training, warmup_frames, optimizer, nr_episodes = 10):
		for _ in tqdm(range(nr_episodes)):
			for frames, labels in dataloader_training:
				prediction = self.single_forward(frames, warmup_frames)
				print(prediction.shape)
				label_tensor = labels.clone().detach()
				targets_one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=11).transpose(1,0)

				loss = torch.nn.functional.smooth_l1_loss(prediction,targets_one_hot)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			print(loss.item())

            

if __name__ =='__main__':
	data = DVSGesture('data/dvs_gesture/', split='testing', preprocessing='stack')
	dataloader_training = DataLoader(data, 2,shuffle=True)
	model = Conv_SNN()
	
	torch.save(model.state_dict(), 'neurobench/examples/model_data/DVS_SNN_untrained.pth')

	optimizer = torch.optim.Adam(model.parameters(),lr=1.2e-3,betas=[0.9,0.99])
	model.fit(dataloader_training=dataloader_training,warmup_frames=70, optimizer=optimizer, nr_episodes=1000)