from neurobench.models import SNNTorchModel, TorchModel
import snntorch
from neurobench.metrics.workload import (
    ClassificationAccuracy,
    MSE,
    SMAPE,
    R2,
    ActivationSparsity,
    SynapticOperations,
    MembraneUpdates,
    ActivationSparsityByLayer,
)
from neurobench.metrics.static import (
    Footprint,
    ParameterCount,
    ConnectionSparsity,
)
import tests.models.model_list as models
import torch.nn as nn
import unittest
import torch


# Pytest for footprint from benchmarks/static_metrics


class TestStaticMetrics(unittest.TestCase):
    def setUp(self):
        self.dummy_net = nn.Module()

        self.net = models.net

        self.net_rnn = models.net_rnn
        self.footprint = Footprint()
        self.parameter_count = ParameterCount()
        self.connection_sparsity = ConnectionSparsity()

    def test_footprint(self):
        model = SNNTorchModel(self.net)
        if snntorch.__version__ == "0.7.0":
            self.assertEqual(self.footprint(model), 583900)
        else:
            self.assertEqual(self.footprint(model), 1406172)

    def test_parameter_count(self):
        model = SNNTorchModel(self.net)
        self.assertEqual(self.parameter_count(model), 145955)

    def test_connection_sparsity(self):
        # Set all the weights to zero
        for param in self.net.parameters():
            param.data = torch.zeros_like(param.data)
        model = SNNTorchModel(self.net)
        self.assertEqual(self.connection_sparsity(model), 1.0)

        # Set all the weights to one
        for param in self.net.parameters():
            param.data = torch.ones_like(param.data)
        model = SNNTorchModel(self.net)
        self.assertEqual(self.connection_sparsity(model), 0.0)

        # Set all the weights to a random value
        for param in self.net.parameters():
            param.data = torch.rand_like(param.data)
        model = SNNTorchModel(self.net)
        self.assertEqual(self.connection_sparsity(model), 0.0)

        # Set half the weights to zero and half the weights to one
        for param in self.net.parameters():
            param.data[: param.data.shape[0] // 2] = torch.zeros_like(
                param.data[: param.data.shape[0] // 2]
            )
            param.data[param.data.shape[0] // 2 :] = torch.ones_like(
                param.data[param.data.shape[0] // 2 :]
            )
        model = SNNTorchModel(self.net)
        # Assert the connection sparsity is within 0.001 of 0.5
        self.assertLess(abs(self.connection_sparsity(model) - 0.5), 0.001)

        model = SNNTorchModel(self.net_rnn)
        self.assertLess(self.connection_sparsity(model), 0.001)


class TestWorkloadMetrics(unittest.TestCase):
    def setUp(self):
        self.dummy_net = nn.Module()
        self.net = models.net
        self.net_relu_0 = models.net_relu_0
        self.net_relu_0_2 = models.net_relu_0_2
        self.net_relu_50 = models.net_relu_50
        self.net_relu_50_2 = models.net_relu_50_2
        self.net_torch_relu_0 = models.net_torch_relu_0
        self.net_sigm = models.net_sigm
        self.net_conv_2d = models.net_conv_2d
        self.net_conv_1d = models.net_conv_1d
        self.net_snn = models.net_snn
        self.net_RNN = models.simple_RNN()
        self.net_GRU = models.simple_GRU()
        self.net_lstm = models.simple_LSTM()

        self.classification_accuracy = ClassificationAccuracy()
        self.mse = MSE()
        self.smape = SMAPE()
        self.r2 = R2()
        self.activation_sparsity = ActivationSparsity()
        self.synaptic_operations = SynapticOperations()
        self.mem_updates = MembraneUpdates()
        self.activation_sparsity_by_layer = ActivationSparsityByLayer()

    def test_classification_accuracy(self):
        model = SNNTorchModel(self.dummy_net)

        batch_size = 10
        data = (
            torch.randn(batch_size),
            torch.arange(1, batch_size + 1),
        )

        # all wrong
        preds = torch.zeros(batch_size)
        self.assertEqual(
            round(self.classification_accuracy(model, preds, data), 1), 0.0
        )

        # one correct
        preds = torch.ones(batch_size)
        self.assertEqual(
            round(self.classification_accuracy(model, preds, data), 1), 0.1
        )

        # half correct
        preds = torch.tensor([0 if i % 2 == 0 else i + 1 for i in range(batch_size)])
        self.assertEqual(
            round(self.classification_accuracy(model, preds, data), 1), 0.5
        )

        # all correct
        preds = torch.arange(1, batch_size + 1)
        self.assertEqual(
            round(self.classification_accuracy(model, preds, data), 1), 1.0
        )

    def test_MSE(self):
        # dummy model
        model = SNNTorchModel(self.dummy_net)

        batch_size = 10
        data = (
            torch.randn(batch_size),
            torch.arange(0, batch_size),
        )  # input and targets

        preds = torch.tensor([1.1 * i for i in range(batch_size)])

        diff = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        diff = [i**2 for i in diff]
        correct = sum(diff) / batch_size

        self.assertEqual(round(self.mse(model, preds, data), 3), round(correct, 3))

    def test_sMAPE(self):
        # dummy model
        model = SNNTorchModel(self.dummy_net)

        batch_size = 10
        data = (
            torch.randn(batch_size),
            torch.arange(1, batch_size + 1),
        )  # input and targets

        preds = torch.tensor([1.1 * (i + 1) for i in range(batch_size)])

        diff = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        added = [i + 1.1 * i for i in range(1, batch_size + 1)]

        diff = [i / j for i, j in zip(diff, added)]
        correct = sum(diff) * 200 / batch_size

        self.assertEqual(round(self.smape(model, preds, data), 3), round(correct, 3))

    def test_r2(self):
        # dummy model
        model = SNNTorchModel(self.dummy_net)

        batch_size = 10
        targets = [[i for i in range(batch_size)] for i in range(2)]

        data = (
            torch.randn(2, batch_size),
            torch.tensor(targets, dtype=torch.float).transpose(0, 1),
        )  # input and targets

        preds = [
            [1.1 * i for i in range(batch_size)],
            [1.1 * (i + 1) for i in range(batch_size)],
        ]

        preds = torch.tensor(preds).transpose(0, 1)

        x_diff = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        y_diff = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        x_num = sum([i**2 for i in x_diff])
        y_num = sum([i**2 for i in y_diff])

        x_den = 8.25 * batch_size
        y_den = 8.25 * batch_size

        x_r2 = 1 - x_num / x_den
        y_r2 = 1 - y_num / y_den
        correct = (x_r2 + y_r2) / 2

        self.assertEqual(round(self.r2(model, preds, data), 3), round(correct, 3))

    def test_activation_sparsity(self):
        # test spiking model
        model = SNNTorchModel(self.net)
        # detect_activations_connections(model)
        model.register_hooks()
        self.assertEqual(len(model.activation_hooks), 4)

        # test ReLU model

        model_relu_0 = TorchModel(self.net_relu_0)
        # detect_activations_connections(model_relu_0)
        model_relu_0.register_hooks()
        inp = torch.ones(20)

        out_relu = model_relu_0(inp)
        act_sp_relu_0 = self.activation_sparsity(model_relu_0, out_relu, inp)
        self.assertEqual(act_sp_relu_0, 0.0)

        # test ReLU model with half-negative inputs
        inp = torch.ones(20)
        inp[0:10] = -1

        model_relu_50 = TorchModel(self.net_relu_50)
        model_relu_50.register_hooks()
        out_relu_50 = model_relu_50(inp)

        act_sp_relu_50 = self.activation_sparsity(model_relu_50, out_relu_50, inp)

        self.assertEqual(act_sp_relu_50, 0.5)

        # test duplicating activation layers in the model

        model_torch_relu_0 = TorchModel(self.net_torch_relu_0)
        model_torch_relu_0.register_hooks()
        inp = torch.ones(20)

        out_relu = model_torch_relu_0(inp)
        act_sp_torch_relu_0 = self.activation_sparsity(
            model_torch_relu_0, out_relu, inp
        )

        self.assertEqual(act_sp_torch_relu_0, 0.0)

        model_torch_relu_0.reset_hooks()
        inp = torch.ones(20)
        inp[0:10] = -1
        out_relu = model_torch_relu_0(inp)
        act_sp_relu_50 = self.activation_sparsity(model_torch_relu_0, out_relu, inp)

        self.assertEqual(act_sp_relu_50, 0.5)

        # test Sigmoid model

        model_sigm = TorchModel(self.net_sigm)
        model_sigm.register_hooks()

        inp = torch.ones(20)
        out_sigm = model_sigm(inp)
        act_sp_sigm = self.activation_sparsity(model_sigm, out_sigm, inp)
        self.assertEqual(act_sp_sigm, 0.0)

    def test_synaptic_ops(self):
        # test ReLU model

        model_relu_0 = TorchModel(self.net_relu_0_2)
        model_relu_0.register_hooks()
        inp = torch.ones(1, 20)
        inp[:, 0:10] = 5

        out_relu = model_relu_0(inp)
        syn_ops = self.synaptic_operations(model_relu_0, out_relu, (inp, 0))
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 1125)
        self.assertEqual(syn_ops["Effective_ACs"], 0)

        # test model with Identity layer as first layer

        inp = torch.ones(1, 20)
        inp[:, 0:10] = 5

        model_relu_50 = TorchModel(self.net_relu_50_2)
        model_relu_50.register_hooks()
        out_relu_50 = model_relu_50(inp)

        syn_ops = self.synaptic_operations(model_relu_50, out_relu_50, (inp, 0))
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], (2 * 625 + 400 + 500))
        self.assertEqual(syn_ops["Effective_ACs"], 0)

        # test conv2d layers
        inp = torch.ones(1, 1, 3, 3)  # 9 syn ops
        inp[0, 0, 0, 0] = 4  # avoid getting classified as snn

        model = TorchModel(self.net_conv_2d)
        model.register_hooks()

        out = model(inp)
        syn_ops = self.synaptic_operations(model, out, (inp, 0))
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 9)
        self.assertEqual(syn_ops["Effective_ACs"], 0)

        model.reset_hooks()
        inp = torch.ones(
            1, 1, 12, 12
        )  # (12-(kernelsize-1))**2 * 9 synops per kernel ops= 100*9 syn ops = 900
        inp[0, 0, 0, 0] = 4  # avoid getting classified as snn

        out = model(inp)
        syn_ops = self.synaptic_operations(model, out, (inp, 0))
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 900)
        self.assertEqual(syn_ops["Effective_ACs"], 0)

        inp = torch.ones(1, 5, 10)  # 5*5*(10-(5-1)) = 150 syn ops
        inp[0, 0, 0] = 4  # avoid getting classified as snn

        model = TorchModel(self.net_conv_1d)

        model.register_hooks()

        out = model(inp)
        syn_ops = self.synaptic_operations(model, out, (inp, 0))
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 150)
        self.assertEqual(syn_ops["Effective_ACs"], 0)

        # test snn layers

        # should be 20*5 = 100 syn ops per call, for timesteps (1 sample per batch): 10*100 = 1000

        # simulate spiking input with only ones
        inp = torch.ones(5, 10, 20)  # batch size, time steps, input size

        model = SNNTorchModel(self.net_snn)

        model.register_hooks()

        out = model(inp)
        syn_ops = self.synaptic_operations(model, out, (inp, 0))
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 0)
        self.assertEqual(syn_ops["Effective_ACs"], 1000)

        # test lstm network
        batch_size = 2
        inp = [
            torch.ones(batch_size, 25),
            (torch.ones(batch_size, 5), torch.ones(batch_size, 5)),
        ]  # input (batch_size, inp_size), (hidden, cell)
        inp[0][0, 0] = 4  # avoid getting classified as snn
        model = TorchModel(self.net_lstm)

        model.register_hooks()

        out = model(inp)

        syn_ops = self.synaptic_operations(model, out, inp)
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 615)
        self.assertEqual(syn_ops["Effective_ACs"], 0)

        # test RNN network
        batch_size = 2
        inp = [
            torch.ones(batch_size, 25),
            torch.ones(batch_size, 5),
        ]  # input, (hidden, cell)
        inp[0][0, 0] = 4  # avoid getting classified as snn
        model = TorchModel(self.net_RNN)

        model.register_hooks()

        out = model(inp)

        syn_ops = self.synaptic_operations(model, out, inp)
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 150)
        self.assertEqual(syn_ops["Effective_ACs"], 0)

        # test GRU network
        batch_size = 2
        inp = [
            torch.ones(batch_size, 25),
            torch.ones(batch_size, 5),
        ]  # input, (hidden, cell)
        inp[0][0, 0] = 4  # avoid getting classified as snn
        model = TorchModel(self.net_GRU)

        model.register_hooks()

        out = model(inp)

        syn_ops = self.synaptic_operations(model, out, inp)
        self.synaptic_operations.reset()

        self.assertEqual(syn_ops["Effective_MACs"], 465)
        self.assertEqual(syn_ops["Effective_ACs"], 0)

    def test_membrane_potential_updates(self):
        # simulate spiking input with only ones
        inp = torch.ones(5, 10, 20)  # batch size, time steps, input size

        model = SNNTorchModel(self.net_snn)

        model.register_hooks()

        out = model(inp)
        tot_mem_updates = self.mem_updates(model, out, (inp, 0))
        self.mem_updates.reset()

        self.assertEqual(tot_mem_updates, 50)

    def test_activation_sparsity_by_layer(self):

        inp = torch.ones(5, 10, 20)  # batch size, time steps, input size

        model = SNNTorchModel(self.net_snn)

        model.register_hooks()

        out = model(inp)
        act_sparsity_by_layer = self.activation_sparsity_by_layer(model, out, (inp, 0))
        self.activation_sparsity_by_layer.reset()

        self.assertEqual(act_sparsity_by_layer["1"], 0.96)


# TODO: refactor this metric if needed
# def test_neuron_update_metric():
#     net_relu_0 = nn.Sequential(
#         # nn.Flatten(),
#         nn.Linear(20, 25, bias=False),
#         nn.Sigmoid(),
#         nn.Linear(25, 25, bias=False),
#         nn.ReLU(),
#         nn.Linear(25, 25, bias=False),
#         nn.Sigmoid(),
#         nn.Linear(25, 25, bias=False),
#         nn.ReLU(),
#         nn.Linear(25, 25, bias=False),
#         snn.Lapicque(
#             beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True, output=True
#         ),
#     )
#     model_relu_0 = SNNTorchModel(net_relu_0)
#     detect_activations_connections(model_relu_0)
#     inp = torch.ones(1, 1, 20)
#     out_relu = model_relu_0(inp)
#     neuron_updates = number_neuron_updates(model_relu_0, out_relu, (inp, 0))
#     print(neuron_updates)
#     print("Manual check!")
#     print("Passed neuron update metric")
#
#
#
# test_neuron_update_metric()
