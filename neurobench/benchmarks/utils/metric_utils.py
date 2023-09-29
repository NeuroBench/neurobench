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
	in_states = True # assume that input is tuple of inputs and states. If not, then set to False
	spiking = False
	with torch.no_grad():
		inps = []
		if isinstance(inputs, tuple):
				# input is first element, rest is hidden states
				test_ins = inputs[0]

				if len(test_ins[(test_ins != 0) & (test_ins !=1)])==0:
					spiking=True
				for inp in inputs:
					if inp is not None:
						if isinstance(inp, tuple): # these are the states
							nps = []
							for np in nps:
								if np is not None:
									np[np != 0] = 1
									inps.append(np)
						else:
							if inp is not None:
								inp[inp != 0] = 1
								inps.append(inp)
						
		else:
			in_states = False
			if len(inputs[(inputs != 0) & (inputs !=1)])==0:
				spiking = True

			inputs[inputs != 0] = 1
			inps.append(inputs)

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
		
		nr_updates = layer_bin(inputs) # this returns the number of MACs for every output neuron: if spiking neurons only AC
		macs = nr_updates.sum() + add_bias # returns total macs

	elif isinstance(layer, recurrent_layers):
		layer_bin = make_binary_copy(layer)
		attribute_names = []
		for i in range(layer.num_layers): 
			param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
			if layer.bias:
				param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
			if layer.proj_size > 0: # it is lstm
				param_names += ['weight_hr_l{}{}']

			attribute_names += [x.format(i, '') for x in param_names]
			if layer.bidirectional:
				suffix = '_reverse'
				attribute_names += [x.format(i, suffix) for x in param_names]
		raise 'This layer is not yet supported by NeuroBench.'
		return 0
	elif isinstance(layer, recurrent_cells):
		# NOTE: sigmoid and tanh will never change a non-zero value to zero or vice versa
		# NOTE: these activation functions are currently NOT included in NeuroBench
		# if no explicit states are passed to recurrent layers, then h and c are initialized to zero (pytorch convention)
		layer_bin = make_binary_copy(layer)
		# transpose from batches, timesteps, features to features, batches
		# print(layer_bin.weight_ih.shape)
		out_ih = torch.matmul(layer_bin.weight_ih, inputs[0].transpose(0,-1)) # accounts for i,f,g,o
		out_hh = torch.zeros_like(out_ih)
		
		
		biases = 0
		bias_ih = 0
		bias_hh = 0
		# out matrices are now features, batches
		if layer_bin.bias:
			bias_ih = layer_bin.bias_ih.unsqueeze(0).transpose(0,-1)
			bias_hh = layer_bin.bias_hh.unsqueeze(0).transpose(0,-1)
			biases = bias_ih + bias_hh
		
		if isinstance(layer, torch.nn.LSTMCell):
			# number of operations for lstmcells
			# i = sigmoid(Wii*x + bii + Whi*h + bhi) 
			# f = sigmoid(Wif*x + bif + Whf*h + bhf)
			# g = tanh(Wig*x + big + Whg*h + bhg)
			# o = sigmoid(Wio*x + bio + Who*h + bho)

			# c = f*c + i*g
			# h = o*tanh(c)

			# inputs = (x,(h,c))
			if in_states:
				out_hh = torch.matmul(layer_bin.weight_hh, inputs[1][0].transpose(0,-1))
			out = out_ih + out_hh + biases
			ifgo_macs = out.sum() # accounts for i,f,g,o

			out[out!=0] = 1
			# out is vector with i,f,g,o
			ifgo = out.reshape(4,-1) # each row is one of i,f,g,o
			if in_states:
				c_1 = ifgo[1,:]*inputs[1][1] + ifgo[0,:]*ifgo[2,:]
			else:
				c_1 = ifgo[0,:]*ifgo[2,:]
			c_1[c_1!=0] = 1
			ifgoc_macs = ifgo_macs + c_1.sum()
			output = ifgo[3,:]*c_1 # drop tanh as does not affect 1 vs 0
			output[output!=0] = 1
			macs = output.sum() + ifgoc_macs
		
		if isinstance(layer, torch.nn.RNNCell):
			if in_states:
				out_hh = torch.matmul(layer_bin.weight_hh, inputs[1].transpose(0,-1))
			out = out_ih + out_hh + biases
			macs = out.sum()

		if isinstance(layer, torch.nn.GRUCell):
			if in_states:
				out_hh = torch.matmul(layer_bin.weight_hh, inputs[1].transpose(0,-1))
			# out is vector with r,z,n
			out_ih = out_ih + bias_ih
			out_hh = out_hh + bias_hh
			
			out_ih = out_ih.reshape(3,-1)
			out_hh = out_hh.reshape(3,-1)

			out_ih_n = out_ih[2,:]
			out_hh_n = out_hh[2,:]
			rzn = out_ih + out_hh

			macs += out_ih[0:2,:].sum() + out_hh[0:2,:].sum()
			# rzn = out.reshape(3,-1)
			r = rzn[0,:]
			z = rzn[1,:]
			r[r!=0] = 1
			n = out_ih_n + r*out_hh_n # add 
			macs += n.sum()
			print(macs)
			n[n!=0] = 1
			z_a = (1-z)
			macs += torch.tensor(z.size()).sum() # append number of subtractions
			z_a[z_a!=0] = 1
			t_1 = z_a * n
			t_2 = z * inputs[1]

			out_nrs = t_1 + t_2
			macs += out_nrs.sum()

	return int(macs), spiking
		

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
