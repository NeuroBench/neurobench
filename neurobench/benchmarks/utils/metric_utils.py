import torch
import copy
import torch.profiler as profiler

import copy
import snntorch as snn
from torch import nn

def activation_modules():
    """
    The activation layers that can be auto-deteced. Every activation layer can only be included once.
    """
    return list(set([nn.ReLU,
            nn.Sigmoid, 
           ]))


def check_shape(preds, labels):
	""" Checks that the shape of the predictions and labels are the same.
	"""
	if preds.shape != labels.shape:
		raise ValueError("preds and labels must have the same shape")

def make_binary_copy(layer):
	""" Makes a binary copy of the layer. All non 0 entries are made 1.
	"""
	layer_copy = copy.deepcopy(layer)

	stateless_layers = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)
	recurrent_layers = (torch.nn.RNNBase)
	recurrent_cells  = (torch.nn.RNNCellBase)


	if isinstance(layer, stateless_layers):
		weights = layer_copy.weight.data
		weights[weights != 0] = int(1)

		if layer.bias is not None:
			biases = layer_copy.bias.data
			biases[biases != 0] = int(1)
			layer_copy.bias.data = biases	

		layer_copy.weight.data = weights


	elif isinstance(layer,recurrent_cells):
		attribute_names = ['weight_ih', 'weight_hh']
		if layer.bias:
			attribute_names += ['bias_ih', 'bias_hh']
		# if layer.proj_size > 0: # it is lstm
		# 	attribute_names += ['weight_hr']

		for attr in attribute_names:
			with torch.no_grad():
				attr_val = getattr(layer_copy, attr)
				attr_val[attr_val != 0] = int(1)
				setattr(layer_copy, attr, attr_val)

	return layer_copy


def single_layer_MACs(inputs, layer):
	""" Computes the MACs for a single layer.
	"""
	macs = 0

	with torch.no_grad():
		if isinstance(inputs, tuple):
				print(inputs)
		else:
			inputs[inputs != 0] = 1
	stateless_layers = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)
	recurrent_layers = (torch.nn.RNNBase)
	recurrent_cells  = (torch.nn.RNNCellBase)

	if isinstance(layer, stateless_layers):
		# then multiply the binary layer with the diagonal matrix to get the MACs
		layer_bin = make_binary_copy(layer)

		# how many biases are added
		# if there is a bias
		add_bias = 0
		if layer.bias is not None:
			add_bias = torch.count_nonzero(layer.bias.data)
		
		nr_updates = layer_bin(inputs)
		macs = nr_updates.sum() + add_bias # returns total macs

	elif isinstance(layer, recurrent_layers):
		print("I am a recurrent layer")
		return 0
	elif isinstance(layer, recurrent_cells):
		# number of operations
		# i = sigmoid(Wii*x + bii + Whi*h + bhi) 
		# f = sigmoid(Wif*x + bif + Whf*h + bhf)
		# g = tanh(Wig*x + big + Whg*h + bhg)
		# o = sigmoid(Wio*x + bio + Who*h + bho)

		# c = f*c + i*g
		# h = o*tanh(c)

		# inputs = (x,(h,c))


		# NOTE: sigmoid and tanh will never change a non-zero value to zero or vice versa
		# NOTE: these activation functions are currently NOT included in NeuroBench
		layer_bin = make_binary_copy(layer)
		# transpose from batches, features to features, batches
		out_ih = torch.matmul(layer_bin.weight_ih, inputs[0].transpose(0,-1)) # accounts for i,f,g,o
		out_hh = torch.matmul(layer_bin.weight_hh, inputs[1][0].transpose(0,-1)) # accounts for i,f,g,o
		
		# out matrices are now features, batches
		if layer_bin.bias:
			biases = (layer_bin.bias_ih + layer_bin.bias_hh).unsqueeze(0).transpose(0,-1)

		out = out_ih + out_hh + biases
		ifgo_macs = out.sum() # accounts for i,f,g,o

		out[out!=0] = 1

		# out is vector with i,f,g,o
		print('Check ordering of ifgo in metric_utils.py!')
		ifgo = out.reshape(4,-1) # each row is one of i,f,g,o
		c_1 = ifgo[1,:]*inputs[1][1] + ifgo[0,:]*ifgo[2,:]
		c_1[c_1!=0] = 1
		ifgoc_macs = ifgo_macs + c_1.sum()
		output = ifgo[3,:]*c_1 # drop tanh as does not affect 1 vs 0
		output[output!=0] = 1
		macs = output.sum() + ifgoc_macs
		

		print('LSTM not implenting bias or c and h comps')
		# return out
	return int(macs)
		

# if __name__=='__main__':
# 	inputs = torch.tensor([[1., 0., 1., 0., 1]])

# 	layer = torch.nn.Linear(5, 4, bias=False)
# 	layer_conv = torch.nn.Conv1d(1, 1, 3, bias=False)
# 	net = torch.nn.Sequential(layer)
# 	net_conv = torch.nn.Sequential(layer_conv)
# 	single_layer_MACs(inputs, layer)

# 	single_layer_MACs(inputs, layer_conv)
# 	with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU],
#     on_trace_ready=None,
# 	with_flops=True,
# 	) as prof:
# 		output = net_conv(inputs)
# 	# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
