"""
Tests for the Neuromorphic Sequential Arena (NSA) examples under examples/nsa.

These need the external ``neuroseqbench`` package and are skipped when it is not
installed.
"""

import importlib.util
import os
import sys
import unittest

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("neuroseqbench")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSA_DIR = os.path.join(ROOT, "examples", "nsa")
HAR_DIR = os.path.join(NSA_DIR, "har")


def _load(name, path):
    """Import an example file under a unique module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, NSA_DIR)
sys.path.insert(0, HAR_DIR)
nsa_common = _load("nsa_common", os.path.join(NSA_DIR, "nsa_common.py"))
har_model = _load("nsa_har_model", os.path.join(HAR_DIR, "model.py"))
har_benchmark = _load("nsa_har_benchmark", os.path.join(HAR_DIR, "benchmark.py"))


def _layer_sizes(hidden):
    sizes = [har_model.INPUT_CHANNELS] + list(hidden) + [har_model.NUM_CLASSES]
    return list(zip(sizes[:-1], sizes[1:]))


class TestNSAHAR(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch, self.steps = 8, har_model.TIME_STEPS
        self.x = torch.randn(self.batch, self.steps, har_model.INPUT_CHANNELS) * 2
        self.y = torch.randint(0, har_model.NUM_CLASSES, (self.batch,))

    def test_wrapper_shapes(self):
        net = har_model.build_model()
        mean = nsa_common.TimeFirstWrapper(net, readout="mean")
        last = nsa_common.TimeFirstWrapper(net, readout="last")
        self.assertEqual(mean(self.x).shape, (self.batch, har_model.NUM_CLASSES))
        self.assertEqual(last(self.x).shape, (self.batch, har_model.NUM_CLASSES))
        with self.assertRaises(ValueError):
            nsa_common.TimeFirstWrapper(net, readout="max")

    def test_wrapper_matches_time_first_net(self):
        net = har_model.build_model().eval()
        with torch.no_grad():
            expected = net(self.x.transpose(0, 1)).mean(0)
            got = nsa_common.TimeFirstWrapper(net)(self.x)
        torch.testing.assert_close(got, expected)

    def test_spike_layers_are_hooked(self):
        model = nsa_common.wrap_for_neurobench(har_model.build_model())
        n_hidden = len(har_model.LIF_FF_CONFIG["hidden_dim"])
        self.assertEqual(len(model.activation_layers()), n_hidden)
        self.assertEqual(len(model.connection_layers()), n_hidden + 1)

    def test_recurrent_last_layer_is_feedforward(self):
        net = har_model.build_model(recurrent=True)
        self.assertTrue(net.spk0.recurrent)
        self.assertTrue(net.spk1.recurrent)
        self.assertFalse(net.spk2.recurrent)

    def test_load_raw_nsa_state_dict(self):
        ref = har_model.build_model(recurrent=True, hidden_dim=[16, 24, 32]).eval()
        path = os.path.join(self._tmpdir(), "nsa_raw.pt")
        torch.save({"state_dict": ref.state_dict()}, path)
        loaded = har_model.load_model(path)
        with torch.no_grad():
            torch.testing.assert_close(
                loaded(self.x.transpose(0, 1)), ref(self.x.transpose(0, 1))
            )

    def test_benchmark_metrics(self):
        hidden = [16, 32, 32]
        net = har_model.build_model(hidden_dim=hidden).eval()
        loader = DataLoader(TensorDataset(self.x, self.y), batch_size=4)
        results = har_benchmark.build_benchmark(net, loader).run(quiet=True)

        n_params = sum(i * o + o for i, o in _layer_sizes(hidden))
        dense = sum(i * o for i, o in _layer_sizes(hidden)) * self.steps
        input_macs = har_model.INPUT_CHANNELS * hidden[0] * self.steps

        self.assertEqual(results["ParameterCount"], n_params)
        self.assertEqual(results["SynapticOperations"]["Dense"], dense)
        # Only the first layer sees real-valued input, everything after is spikes.
        self.assertEqual(results["SynapticOperations"]["Effective_MACs"], input_macs)
        self.assertTrue(0.0 <= results["ActivationSparsity"] <= 1.0)
        self.assertTrue(0.0 <= results["ClassificationAccuracy"] <= 1.0)

    def _tmpdir(self):
        import tempfile

        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


if __name__ == "__main__":
    unittest.main()
