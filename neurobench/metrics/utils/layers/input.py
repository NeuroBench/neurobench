import torch


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
