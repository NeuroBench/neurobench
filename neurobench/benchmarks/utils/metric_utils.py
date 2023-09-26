import torch
import torch.profiler as profiler

import copy
import torch.profiler as profiler

import copy

def check_shape(preds, labels):
	""" Checks that the shape of the predictions and labels are the same.
	"""
	if preds.shape != labels.shape:
		raise ValueError("preds and labels must have the same shape")

def make_binary_copy(layer):
	""" Makes a binary copy of the layer. All non 0 entries are made 1.
	"""
	layer_copy = copy.deepcopy(layer)
	weights = layer_copy.weight.data
	weights[weights != 0] = int(1)
	layer_copy.weight.data = weights	
	return layer_copy


def single_layer_MACs(input, layer, return_updates=False):
	""" Computes the MACs for a single layer.
	"""
	macs = 0
	# first create matrix with input entries on diagonal
	input[input!= 0] = 1
	# input = input.to(dtype=torch.int8)
	# diag = torch.diag(input)
	if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Conv1d) or isinstance(layer, torch.nn.Conv3d):
		# then multiply the binary layer with the diagonal matrix to get the MACs
		layer_bin = make_binary_copy(layer)
		# how many biases are added
		# if there is a bias
		add_bias = 0
		if layer.bias is not None:
			add_bias = torch.count_nonzero(layer.bias.data)
		
		nr_updates = layer_bin(input)
		macs = nr_updates.sum() + add_bias # returns total macs
		if return_updates:
			return int(macs), torch.count_nonzero(nr_updates)
	elif isinstance(layer, torch.nn.Identity):
		out = layer(input)
		print(out)
		macs = out.sum()
		if return_updates:
			return int(macs), torch.count_nonzero(out)
	return int(macs)
		

# if __name__=='__main__':
# 	input = torch.tensor([[1., 0., 1., 0., 1]])

# 	layer = torch.nn.Linear(5, 4, bias=False)
# 	layer_conv = torch.nn.Conv1d(1, 1, 3, bias=False)
# 	net = torch.nn.Sequential(layer)
# 	net_conv = torch.nn.Sequential(layer_conv)
# 	single_layer_MACs(input, layer)

# 	single_layer_MACs(input, layer_conv)
# 	with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU],
#     on_trace_ready=None,
# 	with_flops=True,
# 	) as prof:
# 		output = net_conv(input)
# 	# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
