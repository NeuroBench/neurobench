import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate as surrogate
from neurobench.models import SNNTorchModel
from neurobench.benchmarks.static_metrics import model_size, parameter_count, connection_sparsity
from neurobench.benchmarks.data_metrics import classification_accuracy, MSE, sMAPE, r2


# Pytest for model_size from benchmarks/static_metrics
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

# Pytest for parameter_count from benchmarks/static_metrics
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

# Pytest for connection_sparsity from benchmarks/static_metrics
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

#Pytest for classification_accuracy from benchmarks/data_metrics
def test_classification_accuracy():
    # dummy model
    model = SNNTorchModel(net = nn.Module())

    batch_size = 10
    data = (torch.randn(batch_size), torch.arange(1, batch_size+1)) # input and targets

    # all wrong
    preds = torch.zeros(batch_size)
    assert round(classification_accuracy(model, preds, data), 1) == 0.0

    # one correct
    preds = torch.ones(batch_size)
    assert round(classification_accuracy(model, preds, data), 1) == 0.1

    # half correct
    preds = torch.tensor([0 if i%2==0 else i+1 for i in range(batch_size)])
    assert round(classification_accuracy(model, preds, data), 1) == 0.5

    # all correct
    preds = torch.arange(1, batch_size+1)
    assert round(classification_accuracy(model, preds, data), 1) == 1.0

#Pytest for MSE from benchmarks/data_metrics
def test_MSE():
    # dummy model
    model = SNNTorchModel(net = nn.Module())

    batch_size = 10
    data = (torch.randn(batch_size), torch.arange(0, batch_size)) # input and targets

    preds = torch.tensor([1.1 * i for i in range(batch_size)])

    diff = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9]
    diff = [i**2 for i in diff]
    correct = sum(diff) / batch_size

    assert round(MSE(model, preds, data), 3) == round(correct, 3)

#Pytest for sMAPE from benchmarks/data_metrics
def test_sMAPE():
    # dummy model
    model = SNNTorchModel(net = nn.Module())

    batch_size = 10
    data = (torch.randn(batch_size), torch.arange(1, batch_size+1)) # input and targets

    preds = torch.tensor([1.1 * (i+1) for i in range(batch_size)])

    diff = [0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0]
    added = [i + 1.1*i for i in range(1, batch_size+1)]

    diff = [i/j for i, j in zip(diff, added)]
    correct = sum(diff) * 200 / batch_size

    assert round(sMAPE(model, preds, data), 3) == round(correct, 3)

#Pytest for r2 from benchmarks/data_metrics
def test_r2():
    # dummy model
    model = SNNTorchModel(net = nn.Module())

    batch_size = 10
    targets = [[i for i in range(batch_size)] for i in range(2)]

    data = (torch.randn(2, batch_size), torch.tensor(targets).transpose(0, 1)) # input and targets

    preds = [[1.1 * i for i in range(batch_size)], [1.1 * (i+1) for i in range(batch_size)]]

    preds = torch.tensor(preds).transpose(0, 1)

    x_diff = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9]
    y_diff = [1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.7, 1.8, 1.9, 2.0]
    x_num = sum([i**2 for i in x_diff])
    y_num = sum([i**2 for i in y_diff])

    x_den = 8.25*batch_size
    y_den = 8.25*batch_size

    x_r2 = 1 - x_num/x_den
    y_r2 = 1 - y_num/y_den
    correct = (x_r2 + y_r2) / 2

    R2 = r2()
    assert round(R2(model, preds, data), 3) == round(correct, 3)