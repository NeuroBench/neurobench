import torch.nn as nn
import torch

from torch.profiler import profile, record_function, ProfilerActivity

from fvcore.nn import FlopCountAnalysis, flop_count_table

class TestModel(nn.Module):
    '''
    standard ANN, this works fine
    '''
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

class TestSNNModel(nn.Module):
    '''
    Default SNNTorch Leaky layer is not supported by the fvcore tool.

        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/snntorch/_neurons/leaky.py", line 197, in forward
            self.reset = self.mem_reset(self.mem)
        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/snntorch/_neurons/neurons.py", line 106, in mem_reset
            mem_shift = mem - self.threshold
        RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient
    '''

    def __init__(self):
        super().__init__()

        import snntorch as snn
        from snntorch import surrogate

        beta = 0.9
        spike_grad = surrogate.fast_sigmoid()
        self.net = nn.Sequential(
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

    def forward(self, x):
        return self.net(x)

class TestSJModel(nn.Module):
    '''
    Default SJ model also fails fvcore

        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/spikingjelly/activation_based/neuron.py", line 239, in single_step_forward
            self.neuronal_charge(x)
        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/spikingjelly/activation_based/neuron.py", line 729, in neuronal_charge
            self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
        RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient
    '''
    def __init__(self):
        super().__init__()

        from spikingjelly.activation_based import neuron, layer, surrogate

        self.net = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.net(x)

def test(model, inputs):
    model.eval()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            y = model(inputs)

    breakpoint()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))

# TODO: models seem to work fine with the profiler but do we want to simply report the flop count generated here?
# other note: the flop count appears to count adds/mults separately, MACs is half of the FLOPs reported

# test(TestModel(), torch.randn((1,3,10,10)))

# test(TestSNNModel(), torch.randn((1,20)))

test(TestSJModel(), torch.randn((1,28,28)))