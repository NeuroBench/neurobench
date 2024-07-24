from importlib import import_module
import torch
import copy
from torch import nn


def _lazy_import(package_name, module_name, class_name):
    module = import_module(module_name, package=package_name)
    return getattr(module, class_name)


def activation_modules():
    """
    The activation layers that can be auto-deteced.

    Every activation layer can only be included once.

    """
    return list(
        set(
            [
                nn.ReLU,
                nn.Sigmoid,
            ]
        )
    )


def check_shape(preds, labels):
    """Checks that the shape of the predictions and labels are the same."""
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have the same shape")


def make_binary_copy(layer, all_ones=False):
    """
    Makes a binary copy of the layer.

    All non 0 entries are made 1. If all_ones is True, then all entries are made 1.

    """
    layer_copy = copy.deepcopy(layer)

    stateless_layers = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.Conv1d,
        torch.nn.Conv3d,
    )
    # recurrent_layers = (torch.nn.RNNBase)
    recurrent_cells = torch.nn.RNNCellBase

    if isinstance(layer, stateless_layers):
        weights = layer_copy.weight.data
        weights[weights != 0] = int(1)
        if all_ones:
            weights[weights == 0] = int(1)

        if layer.bias is not None:
            biases = layer_copy.bias.data
            biases[biases != 0] = int(1)
            if all_ones:
                biases[biases == 0] = int(1)
            layer_copy.bias.data = biases

        layer_copy.weight.data = weights

    elif isinstance(layer, recurrent_cells):
        attribute_names = ["weight_ih", "weight_hh"]
        if layer.bias:
            attribute_names += ["bias_ih", "bias_hh"]
        # if layer.proj_size > 0: # it is lstm
        # 	attribute_names += ['weight_hr']

        for attr in attribute_names:
            with torch.no_grad():
                attr_val = getattr(layer_copy, attr)
                attr_val[attr_val != 0] = int(1)
                if all_ones:
                    attr_val[attr_val == 0] = int(1)
                setattr(layer_copy, attr, attr_val)

    return layer_copy


def make_ones_copy(layer):
    """
    Makes a ones copy of the layer.

    All entries are made 1.

    """
    layer_copy = copy.deepcopy(layer)

    stateless_layers = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.Conv1d,
        torch.nn.Conv3d,
    )
    # recurrent_layers = (torch.nn.RNNBase)
    recurrent_cells = torch.nn.RNNCellBase

    if isinstance(layer, stateless_layers):
        weights = layer_copy.weight.data
        weights[weights != 0] = int(1)
        weights[weights == 0] = int(1)

        if layer.bias is not None:
            biases = layer_copy.bias.data
            biases[biases != 0] = int(1)
            biases[biases == 0] = int(1)
            layer_copy.bias.data = biases

        layer_copy.weight.data = weights

    elif isinstance(layer, recurrent_cells):
        attribute_names = ["weight_ih", "weight_hh"]
        if layer.bias:
            attribute_names += ["bias_ih", "bias_hh"]
        # if layer.proj_size > 0: # it is lstm
        # 	attribute_names += ['weight_hr']

        for attr in attribute_names:
            with torch.no_grad():
                attr_val = getattr(layer_copy, attr)
                attr_val[attr_val != 0] = int(1)
                attr_val[attr_val == 0] = int(1)
                setattr(layer_copy, attr, attr_val)

    return layer_copy


def cylce_tuple(tup):
    """Returns a copy of the tuple with binary elements."""
    tup_copy = []
    for t in tup:
        if isinstance(t, tuple):
            tup_copy.append(cylce_tuple(t))
        elif t is not None:
            t = t.detach().clone()
            t[t != 0] = 1
            tup_copy.append(t)
    return tuple(tup_copy)


def cylce_tuple_ones(tup):
    """Returns a copy of the tuple with ones elements."""
    tup_copy = []
    for t in tup:
        if isinstance(t, tuple):
            tup_copy.append(cylce_tuple(t))
        elif t is not None:
            t = t.detach().clone()
            t[t != 0] = 1
            t[t == 0] = 1
            tup_copy.append(t)
    return tuple(tup_copy)


def binary_inputs(inputs, all_ones=False):
    """Returns a copy of the inputs with binary elements, all ones if all_ones is
    True."""
    in_states = True  # assume that input is tuple of inputs and states. If not, then set to False
    spiking = False

    with torch.no_grad():
        # TODO: should change this code block so that all inputs get cloned
        if isinstance(inputs, tuple):
            # input is first element, rest is hidden states
            test_ins = inputs[0]

            # NOTE: this only checks first input as everything else can be seen as hidden states in rnn block
            if len(test_ins[(test_ins != 0) & (test_ins != 1) & (test_ins != -1)]) == 0:
                spiking = True
            if not all_ones:
                inputs = cylce_tuple(inputs)
            else:
                inputs = cylce_tuple_ones(inputs)
        else:
            # clone tensor since it may be used as input to other layers
            inputs = inputs.detach().clone()
            in_states = False
            if len(inputs[(inputs != 0) & (inputs != 1) & (inputs != -1)]) == 0:
                spiking = True

            inputs[inputs != 0] = 1
            if all_ones:
                inputs[inputs == 0] = 1
    return inputs, spiking, in_states


def single_layer_MACs(inputs, layer, total=False):
    """
    Computes the MACs for a single layer.

    returns effective operations if total=False, else total operations (including zero operations)
    Supported layers: Linear, Conv1d, Conv2d, Conv3d, RNNCellBase, LSTMCell, GRUCell

    """
    macs = 0

    # copy input
    inputs, spiking, in_states = binary_inputs(inputs, all_ones=total)

    stateless_layers = (
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.Conv1d,
        torch.nn.Conv3d,
    )
    recurrent_layers = torch.nn.RNNBase
    recurrent_cells = torch.nn.RNNCellBase

    if isinstance(layer, stateless_layers):
        # then multiply the binary layer with the diagonal matrix to get the MACs
        layer_bin = make_binary_copy(layer, all_ones=total)

        # bias is not considered as a synaptic operation
        # in the future you can change this parameter to include bias
        bias = False
        if layer_bin.bias is not None and not bias:
            # suppress the bias to zero
            layer_bin.bias.data = torch.zeros_like(layer_bin.bias.data)

        nr_updates = layer_bin(
            inputs
        )  # this returns the number of MACs for every output neuron: if spiking neurons only AC
        macs = nr_updates.sum()

    elif isinstance(layer, recurrent_layers):
        layer_bin = make_binary_copy(layer, all_ones=total)
        attribute_names = []
        for i in range(layer.num_layers):
            param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
            if layer.bias:
                param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
            if layer.proj_size > 0:  # it is lstm
                param_names += ["weight_hr_l{}{}"]

            attribute_names += [x.format(i, "") for x in param_names]
            if layer.bidirectional:
                suffix = "_reverse"
                attribute_names += [x.format(i, suffix) for x in param_names]
        raise "This layer is not yet supported by NeuroBench."
        return 0
    elif isinstance(layer, recurrent_cells):
        # NOTE: sigmoid and tanh will never change a non-zero value to zero or vice versa
        # NOTE: these activation functions are currently NOT included in NeuroBench
        # if no explicit states are passed to recurrent layers, then h and c are initialized to zero (pytorch convention)
        layer_bin = make_binary_copy(layer, all_ones=total)
        # layer_weight_ih is [4*hidden_size, input_size]
        # inputs[0].transpose(0, -1) is [input_size, batch_size]
        out_ih = torch.matmul(
            layer_bin.weight_ih, inputs[0].transpose(0, -1)
        )  # accounts for i,f,g,o
        out_hh = torch.zeros_like(out_ih)
        # out shape is 4*h, batch, for hidden feature dim h

        biases = 0
        bias_ih = 0
        bias_hh = 0
        # out matrices are now features, batches
        if layer_bin.bias:
            bias_ih = layer_bin.bias_ih.unsqueeze(0).transpose(0, -1)
            bias_hh = layer_bin.bias_hh.unsqueeze(0).transpose(0, -1)
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
                out_hh = torch.matmul(
                    layer_bin.weight_hh, inputs[1][0].transpose(0, -1)
                )

            # out_ih[out_ih!=0] = 1
            # out_hh[out_hh!=0] = 1

            out = out_ih + out_hh

            ifgo_macs = out.sum()  # accounts for i,f,g,o WITHOUT biases

            out += biases  # biases are added here for computation of c and h which depend on correct computation of ifgo
            out[out != 0] = 1

            # out is vector with i,f,g,o, shape is 4*h, batch
            hidden = out.shape[0] // 4
            ifgo = out.reshape(4, hidden, -1)  # 4, h, B
            if in_states:
                # inputs[1][1] shape is [B, h]
                # element-wise multiply (vector products f*c + i*g)
                c_1 = (
                    ifgo[1, :] * inputs[1][1].transpose(0, -1) + ifgo[0, :] * ifgo[2, :]
                )
            else:
                c_1 = ifgo[0, :] * ifgo[2, :]

            ifgoc_macs = ifgo_macs + c_1.sum()

            c_1[c_1 != 0] = 1
            output = ifgo[3, :] * c_1  # drop tanh as does not affect 1 vs 0
            output[output != 0] = 1
            macs = output.sum() + ifgoc_macs

        if isinstance(layer, torch.nn.RNNCell):
            if in_states:
                out_hh = torch.matmul(layer_bin.weight_hh, inputs[1].transpose(0, -1))
            out = out_ih + out_hh  # no biases for synaptic operations
            macs = out.sum()

        if isinstance(layer, torch.nn.GRUCell):
            # r = sigmoid(Wir*x + bir + Whr*h + bhr)
            # z = sigmoid(Wiz*x + biz + Whz*h + bhz)
            # n = tanh(Win*x + bin + r*(Whn*h + bhn))
            # h = (1-z)*n + z*h
            if in_states:
                out_hh = torch.matmul(layer_bin.weight_hh, inputs[1].transpose(0, -1))

            rzn = out_ih + out_hh
            # multiplications of all weights and inputs/hidden states
            # Wir*x, Whr*h, Wiz*x, Whz*h, Win*x, Whn*h
            macs += rzn.sum()  # multiplications of all weights and inputs/hidden states
            rzn += biases  # add biases

            hidden = rzn.shape[0] // 3
            rzn = rzn.reshape(3, hidden, -1)  # 3, h, B
            out_hh = out_hh.reshape(3, hidden, -1)
            bias_hh = bias_hh.reshape(3, hidden, -1)

            out_hh_n = out_hh[2, :] + bias_hh[2, :]
            r = rzn[0, :]  # get r
            z = rzn[1, :]

            r[r != 0] = 1
            out_hh_n[out_hh_n != 0] = 1

            n_hh_term_macs = (
                r * out_hh_n
            )  # elementwise_multiplication to find macs of r*(Whn*h + bhn) specifically
            n_hh_term_macs[n_hh_term_macs != 0] = 1
            macs += n_hh_term_macs.sum()

            # note hh part of n is already binarized, does not influence calculation of macs for n
            n = out_hh[2, :] + bias_ih[2, :] + n_hh_term_macs
            n[n != 0] = 1
            z_a = 1 - z
            # only do this now because affects z_a
            z[z != 0] = 1
            z_a[z_a != 0] = 1
            t_1 = z_a * n
            t_2 = z * inputs[1].transpose(
                0, -1
            )  # inputs are shape [B, h], all else is [h, B]

            t_1[t_1 != 0] = 1
            t_2[t_2 != 0] = 1
            out_nrs = t_1 + t_2
            macs += out_nrs.sum()

    return int(macs), spiking
