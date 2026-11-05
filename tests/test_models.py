from neurobench.models.torch_model import TorchModel
from neurobench.models.snntorch_models import SNNTorchModel

import unittest
import torch

from .models.model_list import SimpleCustomSNN, SimpleStepSNN, simple_RNN, simple_GRU


class TestSNNTorchModel(unittest.TestCase):
    """Tests SNNTorchModel class."""

    def setUp(self):
        """Set up test networks and models."""
        self.custom_net = SimpleCustomSNN()
        self.custom_model = SNNTorchModel(self.custom_net, custom_forward=True)

        self.step_net = SimpleStepSNN()
        self.step_model = SNNTorchModel(self.step_net, custom_forward=False)

    def test_custom_forward_runs(self):
        """Test that custom_forward=True calls net.forward directly without error."""
        # (batch, timesteps, channels=5, length=5) for Conv1d(5, 1, 5)
        batch = torch.randn(1, 50, 5, 5)
        output = self.custom_model(batch)
        self.assertIsInstance(output, torch.Tensor)

    def test_custom_forward_output_shape(self):
        """Test that custom_forward=True returns correct shape (batch, timesteps,
        ...)."""
        batch = torch.randn(1, 50, 5, 5)
        output = self.custom_model(batch)
        # conv1d(5,1,5) on length=5 -> length=1, so output is (batch, timesteps, 1, 1)
        self.assertEqual(output.shape[0], 1)  # batch
        self.assertEqual(output.shape[1], 50)  # timesteps

    def test_step_forward_runs(self):
        """Test that custom_forward=False runs step-by-step without error."""
        # (batch, timesteps, features=10)
        batch = torch.randn(2, 30, 10)
        output = self.step_model(batch)
        self.assertIsInstance(output, torch.Tensor)

    def test_step_forward_output_shape(self):
        """Test that custom_forward=False stacks spikes to (batch, timesteps,
        features)."""
        batch = torch.randn(2, 30, 10)
        output = self.step_model(batch)
        self.assertEqual(output.shape, (2, 30, 5))  # (batch, timesteps, out_features)

    def test_step_forward_output_is_spikes(self):
        """Test that non-custom forward output contains only spike values (0 or 1)."""
        batch = torch.randn(2, 30, 10)
        output = self.step_model(batch)
        unique_values = output.unique()
        for v in unique_values:
            self.assertIn(v.item(), [0.0, 1.0])

    def test_net_is_in_eval_mode(self):
        """Test that the wrapped network is set to eval mode on init."""
        self.assertFalse(self.custom_net.training)
        self.assertFalse(self.step_net.training)

    def test_net_returns_underlying_network(self):
        """Test that __net__ returns the original network."""
        self.assertIs(self.custom_model.__net__(), self.custom_net)
        self.assertIs(self.step_model.__net__(), self.step_net)

    def test_step_forward_resets_state_between_calls(self):
        """Test that two identical inputs produce identical outputs (state is reset)."""
        batch = torch.randn(2, 30, 10)
        output1 = self.step_model(batch)
        output2 = self.step_model(batch)
        self.assertTrue(torch.equal(output1, output2))

    def test_custom_forward_batch_size(self):
        """Test that custom_forward respects different batch sizes."""
        for batch_size in [1, 4, 8]:
            batch = torch.randn(batch_size, 50, 5, 5)
            output = self.custom_model(batch)
            self.assertEqual(output.shape[0], batch_size)


class TestTorchModel(unittest.TestCase):
    """Tests TorchModel class."""

    def setUp(self):
        """Set up test networks and models."""
        self.rnn_net = simple_RNN()
        self.rnn_model = TorchModel(self.rnn_net)

        self.gru_net = simple_GRU()
        self.gru_model = TorchModel(self.gru_net)

        # (batch, features)
        self.x = torch.randn(4, 25)
        # (batch, hidden_size)
        self.hidden = torch.zeros(4, 5)

    def test_rnn_forward_runs(self):
        """Test that the RNN model runs without error."""
        output = self.rnn_model((self.x, self.hidden))
        self.assertIsInstance(output, torch.Tensor)

    def test_rnn_forward_output_shape(self):
        """Test that the RNN model returns the correct output shape."""
        output = self.rnn_model((self.x, self.hidden))
        self.assertEqual(output.shape, (4, 5))

    def test_gru_forward_runs(self):
        """Test that the GRU model runs without error."""
        output = self.gru_model((self.x, self.hidden))
        self.assertIsInstance(output, torch.Tensor)

    def test_gru_forward_output_shape(self):
        """Test that the GRU model returns the correct output shape."""
        output = self.gru_model((self.x, self.hidden))
        self.assertEqual(output.shape, (4, 5))

    def test_rnn_output_is_non_negative(self):
        """Test that ReLU is applied and output contains no negative values."""
        output = self.rnn_model((self.x, self.hidden))
        self.assertTrue((output >= 0).all())

    def test_gru_output_is_non_negative(self):
        """Test that ReLU is applied and output contains no negative values."""
        output = self.gru_model((self.x, self.hidden))
        self.assertTrue((output >= 0).all())

    def test_rnn_net_is_in_eval_mode(self):
        """Test that the wrapped RNN network is set to eval mode on init."""
        self.assertFalse(self.rnn_net.training)

    def test_gru_net_is_in_eval_mode(self):
        """Test that the wrapped GRU network is set to eval mode on init."""
        self.assertFalse(self.gru_net.training)

    def test_rnn_net_returns_underlying_network(self):
        """Test that __net__ returns the original RNN network."""
        self.assertIs(self.rnn_model.__net__(), self.rnn_net)

    def test_gru_net_returns_underlying_network(self):
        """Test that __net__ returns the original GRU network."""
        self.assertIs(self.gru_model.__net__(), self.gru_net)

    def test_rnn_forward_is_deterministic(self):
        """Test that two identical inputs produce identical outputs for RNN."""
        output1 = self.rnn_model((self.x, self.hidden))
        output2 = self.rnn_model((self.x, self.hidden))
        self.assertTrue(torch.equal(output1, output2))

    def test_gru_forward_is_deterministic(self):
        """Test that two identical inputs produce identical outputs for GRU."""
        output1 = self.gru_model((self.x, self.hidden))
        output2 = self.gru_model((self.x, self.hidden))
        self.assertTrue(torch.equal(output1, output2))

    def test_rnn_different_batch_sizes(self):
        """Test that the RNN model handles different batch sizes correctly."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 25)
            hidden = torch.zeros(batch_size, 5)
            output = self.rnn_model((x, hidden))
            self.assertEqual(output.shape, (batch_size, 5))

    def test_gru_different_batch_sizes(self):
        """Test that the GRU model handles different batch sizes correctly."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 25)
            hidden = torch.zeros(batch_size, 5)
            output = self.gru_model((x, hidden))
            self.assertEqual(output.shape, (batch_size, 5))
