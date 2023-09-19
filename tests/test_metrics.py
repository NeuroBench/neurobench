import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate as surrogate
from neurobench.models import SNNTorchModel, TorchModel
from neurobench.benchmarks.metrics import model_size, parameter_count, connection_sparsity, activation_sparsity
import neurobench.benchmarks.metrics as metrics

# Pytest for model_size from benchmarks/metrics
def test_model_size():
    beta = 0.9
    spike_grad = surrogate.fast_sigmoid()
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(20, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 35),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )
    model = SNNTorchModel(net)
    assert model_size(model) == 583900

# Pytest for parameter_count from benchmarks/metrics
def test_parameter_count():
    beta = 0.9
    spike_grad = surrogate.fast_sigmoid()
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(20, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 35),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )
    model = SNNTorchModel(net)
    assert parameter_count(model) == 145955

# Pytest for connection_sparsity from benchmarks/metrics
def test_connection_sparsity():
    beta = 0.9
    spike_grad = surrogate.fast_sigmoid()
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(20, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 35),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )
    # Set all the weights to zero
    for param in net.parameters():
        param.data = torch.zeros_like(param.data)
    model = SNNTorchModel(net)
    assert connection_sparsity(model) == 1.0

    # Set all the weights to one
    for param in net.parameters():
        param.data = torch.ones_like(param.data)
    model = SNNTorchModel(net)
    assert connection_sparsity(model) == 0.0

    # Set all the weights to a random value
    for param in net.parameters():
        param.data = torch.rand_like(param.data)
    model = SNNTorchModel(net)
    assert connection_sparsity(model) == 0.0

    # Set half the weights to zero and half the weights to one
    for param in net.parameters():
        param.data[:param.data.shape[0]//2] = torch.zeros_like(param.data[:param.data.shape[0]//2])
        param.data[param.data.shape[0]//2:] = torch.ones_like(param.data[param.data.shape[0]//2:])
    model = SNNTorchModel(net)
    # Assert the connection sparsity is within 0.001 of 0.5
    assert abs(connection_sparsity(model) - 0.5) < 0.001

def test_activation_sparsity():
    # test spiking model
    # beta = 0.9
    # spike_grad = surrogate.fast_sigmoid()
    # net = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(20, 256),
    #     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    #     nn.Linear(256, 256),
    #     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    #     nn.Linear(256, 256),
    #     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    #     nn.Linear(256, 35),
    #     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    # )
    # model = SNNTorchModel(net)
    # inp = torch.ones(20)

    # out = model(inp)
    # act_sp = activation_sparsity(model, out, inp)

    # test ReLU model
    net_relu_0 = nn.Sequential(
        # nn.Flatten(),
        nn.Identity(),
        nn.ReLU(),
    )
    net_relu_50 = nn.Sequential(
        # nn.Flatten(),
        nn.Identity(),
        nn.ReLU(),
    )
    model_relu_0 = TorchModel(net_relu_0)
    metrics.preprocess(model_relu_0)

    inp = torch.ones(20)

    out_relu = model_relu_0(inp)
    act_sp_relu_0 = activation_sparsity(model_relu_0, out_relu, inp)

    assert act_sp_relu_0 == 0.0


    inp = torch.ones(20)
    inp[0:10] = -1

    model_relu_50 = TorchModel(net_relu_50)
    metrics.preprocess(model_relu_50)
    out_relu_50 = model_relu_50(inp)

    act_sp_relu_50 = activation_sparsity(model_relu_50, out_relu_50, inp)
    print(act_sp_relu_50,out_relu_50)

    assert act_sp_relu_50 == 0.5
   # test Sigmoid model
    net_sigm = nn.Sequential(
        nn.Identity(),
        nn.Sigmoid(),
    )
    model_sigm = TorchModel(net_sigm)
    inp = torch.ones(20)
    print(model_sigm.activation_layers())
    out_sigm = model_sigm(inp)
    act_sp_sigm = activation_sparsity(model_sigm, out_sigm, inp)

test_activation_sparsity()