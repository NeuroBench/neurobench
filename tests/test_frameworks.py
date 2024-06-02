from neurobench.models import SNNTorchModel
import tests.models.model_list as models
from snntorch import surrogate
import snntorch as snn
from torch import nn
import unittest
import torch


class TestSNNTorchModel(unittest.TestCase):
    """Tests SNNTorchModel class."""

    def setUp(self):
        """Set up the test data."""
        self.input_size = 20
        self.output_size = 35
        self.batch_size = 256
        self.timesteps = 1000

        net = models.net
        self.model = SNNTorchModel(net)

    def test_snntorch_framework_model_runtime(self):
        """Test that the SNNTorchModel returns the correct shape."""

        data = torch.rand((self.batch_size, self.timesteps, self.input_size))
        spikes = self.model(data)
        self.assertEqual(data.shape, (self.batch_size, self.timesteps, self.input_size))
        self.assertEqual(
            spikes.shape, (self.batch_size, self.timesteps, self.output_size)
        )

        data = torch.rand((self.batch_size, self.timesteps, 2, 2, 5))
        spikes = self.model(data)
        self.assertEqual(data.shape, (self.batch_size, self.timesteps, 2, 2, 5))
        self.assertEqual(
            spikes.shape, (self.batch_size, self.timesteps, self.output_size)
        )

    def test_snntorch_framework_model_runtime_error(self):
        """Test that the SNNTorchModel raises an error if the input shape is
        incorrect."""
        data = torch.rand((self.batch_size, self.timesteps, 10, 5))
        self.assertRaises(RuntimeError, self.model, data)
