import torch.nn as nn
import torch

from fvcore.nn import FlopCountAnalysis, flop_count_table

class TestModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Linear(in_features=1000, out_features=10)
		self.conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1)
		self.act = nn.ReLU()

	def forward(self, x):
		x = self.conv(x)
		x = self.act(x)
		x = x.flatten(1)
		out = self.fc(x)

		return out

model = TestModel()
inputs = torch.randn((1,3,10,10))

y = model(inputs)
flops = FlopCountAnalysis(model, (inputs,))

print(flop_count_table(flops))

breakpoint()