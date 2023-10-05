import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate as surrogate
from neurobench.models import SNNTorchModel, TorchModel
# from neurobench.benchmarks.static_metrics import model_size, parameter_count, connection_sparsity,
# from neurobench.benchmarks.data_metrics import activation_sparsity, classification_accuracy, MSE, sMAPE, r2, 
from neurobench.models import SNNTorchModel
from neurobench.benchmarks.static_metrics import model_size, parameter_count, connection_sparsity
from neurobench.benchmarks.data_metrics import classification_accuracy, MSE, sMAPE, r2, activation_sparsity, detect_activations_connections, synaptic_operations, number_neuron_updates
from torch.profiler import profile, record_function, ProfilerActivity

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

    net_rnn = nn.Sequential(
        nn.Flatten(),
        nn.RNN(input_size=20, hidden_size = 20, num_layers=2, bidirectional=True),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(20, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 35),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )
    model = SNNTorchModel(net_rnn)
    assert connection_sparsity(model) < 0.001
    print('Passed connection sparsity')


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

def test_activation_sparsity():
    # test spiking model
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
    detect_activations_connections(model)
    assert len(model.activation_hooks) == 4


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
    detect_activations_connections(model_relu_0)
    inp = torch.ones(20)

    out_relu = model_relu_0(inp)
    act_sp_relu_0 = activation_sparsity(model_relu_0, out_relu, inp)

    assert act_sp_relu_0 == 0.0

    # test ReLU model with half negative inputs
    inp = torch.ones(20)
    inp[0:10] = -1

    model_relu_50 = TorchModel(net_relu_50)
    detect_activations_connections(model_relu_50)
    out_relu_50 = model_relu_50(inp)

    act_sp_relu_50 = activation_sparsity(model_relu_50, out_relu_50, inp)

    assert act_sp_relu_50 == 0.5

    # test duplicating activation layers in model
    act = nn.ReLU()
    net_torch_relu_0 = nn.Sequential(
        # nn.Flatten(),
        nn.Identity(),
        act,
        nn.Identity(),
        act,
        nn.Identity(),
        act,
        nn.Identity(),
        act,
    )
    model_torch_relu_0 = TorchModel(net_torch_relu_0)
    detect_activations_connections(model_torch_relu_0)
    inp = torch.ones(20)

    out_relu = model_torch_relu_0(inp)
    act_sp_torch_relu_0 = activation_sparsity(model_torch_relu_0, out_relu, inp)

    assert act_sp_torch_relu_0 == 0.0
    
    model_torch_relu_0.reset_hooks()
    inp = torch.ones(20)
    inp[0:10] = -1
    out_relu = model_torch_relu_0(inp)
    act_sp_relu_50 = activation_sparsity(model_torch_relu_0, out_relu, inp)

    assert act_sp_relu_50 == 0.5

   # test Sigmoid model
    net_sigm = nn.Sequential(
        nn.Identity(),
        nn.Sigmoid(),
    )
    model_sigm = TorchModel(net_sigm)
    detect_activations_connections(model_sigm)

    inp = torch.ones(20)
    out_sigm = model_sigm(inp)
    act_sp_sigm = activation_sparsity(model_sigm, out_sigm, inp)
    assert act_sp_sigm == 0.0
    print('Passed activation sparsity')


def test_synaptic_ops():
    # test ReLU model
    net_relu_0 = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(20,25,bias=False),
        nn.Sigmoid(),
        nn.Linear(25,25,bias=False),
        nn.ReLU(),
    )
    
    model_relu_0 = TorchModel(net_relu_0)
    detect_activations_connections(model_relu_0)
    inp = torch.ones(1,20)
    inp[:,0:10] = 5


    out_relu = model_relu_0(inp)
    syn = synaptic_operations()
    syn_ops = syn(model_relu_0, out_relu,  (inp,0))

    assert syn_ops['MACs'] == 1125
    assert syn_ops['ACs'] == 0

    # test model with Identity layer as first layer
    net_relu_50 = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(20,20, bias=False), # 400 ops
        nn.Sigmoid(),
        nn.Linear(20,25,bias=False), # 500 ops
        nn.Sigmoid(),
        nn.Linear(25,25,bias=False), # 625 ops
        nn.Sigmoid(),
        nn.Linear(25,25,bias=False), # 625 ops
        nn.Sigmoid(),
    )
    inp = torch.ones(1,20)
    inp[:,0:10] = 5

    model_relu_50 = TorchModel(net_relu_50)
    detect_activations_connections(model_relu_50)
    out_relu_50 = model_relu_50(inp)

    syn = synaptic_operations()
    syn_ops = syn(model_relu_50, out_relu_50,  (inp,0))

    assert syn_ops['MACs'] == (2*625 + 400 + 500)
    assert syn_ops['ACs'] == 0

    # test conv2d layers
    net_conv = nn.Sequential(
        nn.Conv2d(1,1,3,bias=False), 
        nn.ReLU(),
    )

    inp = torch.ones(1, 1, 3, 3) # 9 syn ops 
    inp[0,0,0,0] = 4 # avoid getting classified as snn

    model = TorchModel(net_conv)
    detect_activations_connections(model)


    out = model(inp)
    syn = synaptic_operations()
    syn_ops = syn(model, out, (inp,0))
    assert syn_ops['MACs'] == 9
    assert syn_ops['ACs'] == 0

    model.reset_hooks()
    inp = torch.ones(1, 1, 12, 12) # (12-(kernelsize-1))**2 * 9 synops per kernel ops= 100*9 syn ops = 900
    inp[0,0,0,0] = 4 # avoid getting classified as snn

    out = model(inp)
    syn = synaptic_operations()
    syn_ops = syn(model, out, (inp,0))

    assert syn_ops['MACs'] == 900
    assert syn_ops['ACs'] == 0

    # test conv1d layers
    net_conv = nn.Sequential(
        # nn.Flatten(),
        nn.Conv1d(5,1,5,bias=False), 
        nn.ReLU(),
    )
    inp = torch.ones(1, 5, 10) # 5*5*(10-(5-1)) = 150 syn ops
    inp[0,0,0] = 4 # avoid getting classified as snn

    model = TorchModel(net_conv)
    
    detect_activations_connections(model)

    out = model(inp)
    syn = synaptic_operations()
    syn_ops = syn(model, out, (inp,0))

    assert syn_ops['MACs'] == 150
    assert syn_ops['ACs'] == 0

    # test snn layers
    net_snn = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(20,5,bias=False), 
        snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True, output=True),
    )
    # should be 20*5 = 100 syn ops per call, for timesteps (1 sample per batch): 10*100 = 1000 

    # simulate spiking input with only ones
    inp = torch.ones(5, 10, 20) # batch size, time steps, input size 
    

    model = SNNTorchModel(net_snn)
    
    detect_activations_connections(model)

    out = model(inp)
    syn = synaptic_operations()
    syn_ops = syn(model, out,  (inp,0))

    assert syn_ops['MACs'] == 0
    assert syn_ops['ACs'] == 1000

    # test lstm network
    net_lstm = simple_LSTM()

    inp = [torch.ones(1,25), (torch.ones(1,5), torch.ones(1,5))] # input, (hidden, cell)
    inp[0][0,0] = 4 # avoid getting classified as snn
    model = TorchModel(net_lstm)

    detect_activations_connections(model)

    out = model(inp)

    syn = synaptic_operations()
    syn_ops = syn(model, out, inp)

    assert syn_ops['MACs'] == 615
    assert syn_ops['ACs'] == 0

    # test RNN network
    net_RNN = simple_RNN()

    inp = [torch.ones(1,25), torch.ones(1,5)] # input, (hidden, cell)
    inp[0][0,0] = 4 # avoid getting classified as snn
    model = TorchModel(net_RNN)

    detect_activations_connections(model)

    out = model(inp)

    syn = synaptic_operations()
    syn_ops = syn(model, out, inp)

    assert syn_ops['MACs'] == 150
    assert syn_ops['ACs'] == 0

    # test GRU network
    net_GRU = simple_GRU()

    inp = [torch.ones(1,25), torch.ones(1,5)] # input, (hidden, cell)
    inp[0][0,0] = 4 # avoid getting classified as snn
    model = TorchModel(net_GRU)

    detect_activations_connections(model)

    out = model(inp)

    syn = synaptic_operations()
    syn_ops = syn(model, out, inp)

    assert syn_ops['MACs'] == 465
    assert syn_ops['ACs'] == 0

    
    print('Passed synaptic ops')



def test_neuron_update_metric():
    net_relu_0 = nn.Sequential(
        # nn.Flatten(),
        nn.Linear(20,25,bias=False),
        nn.Sigmoid(),
        nn.Linear(25,25,bias=False),
        nn.ReLU(),
        nn.Linear(25,25,bias=False),
        nn.Sigmoid(),
        nn.Linear(25,25,bias=False),
        nn.ReLU(),
        nn.Linear(25,25, bias=False),
        snn.Lapicque(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True, output=True),
    )
    model_relu_0 = SNNTorchModel(net_relu_0)
    detect_activations_connections(model_relu_0)
    inp = torch.ones(1,1,20)
    out_relu = model_relu_0(inp)
    neuron_updates = number_neuron_updates(model_relu_0, out_relu, (inp,0))
    print('Manual check!')
    print('Passed neuron update metric')


class simple_LSTM(nn.Module):
    """Nonsense LSTM for operations testing
    Should be 615 MACs
    """
    def __init__(self):
        super(simple_LSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size=25, hidden_size=5, bias=True)
        self.rel = nn.ReLU()
    def forward(self, inp):
        x, states = inp[0], inp[1]
        x, _ = self.lstm(x, states)
        # x = self.rel(x)
        return x

class simple_RNN(nn.Module):
    """Nonsense RNN for operations testing
    Should be 150 MACs
    """
    def __init__(self):
        super(simple_RNN, self).__init__()
        self.RNN = nn.RNNCell(input_size=25, hidden_size=5, bias=True)
        self.rel = nn.ReLU()
    def forward(self, inp):
        x, states = inp[0], inp[1]
        x = self.RNN(x, states)
        x = self.rel(x)
        return x
    
class simple_GRU(nn.Module):
    """Nonsense GRU/RNN for operations testing
    Should be 465 MACs
    """
    def __init__(self):
        super(simple_GRU, self).__init__()
        self.GRU = nn.GRUCell(input_size=25, hidden_size=5, bias=True)
        self.rel = nn.ReLU()
    def forward(self, inp):
        x, states = inp[0], inp[1]
        x = self.GRU(x, states)
        x = self.rel(x)
        return x
    

test_connection_sparsity()
test_activation_sparsity()
test_synaptic_ops()
test_neuron_update_metric()
