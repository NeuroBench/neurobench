from neurobench.preprocessing.speech2spikes import S2SPreProcessor
from pathlib import Path
import torchaudio
import unittest
import torch


class TestS2SPreProcessor(unittest.TestCase):
    """Tests S2SPreProcessor class."""

    def setUp(self):
        """Set up the test data."""
        sample_file = Path(__file__).parent.joinpath("test_data/sample_audio.wav")
        self.sample_audio, _ = torchaudio.load(sample_file)
        self.s2s = S2SPreProcessor()

    def test_initialization(self):
        """Test the initialization of S2SPreProcessor."""
        self.assertEqual(self.s2s.device, None)
        self.assertTrue(self.s2s.transpose)
        self.assertEqual(self.s2s.log_offset, 1e-6)
        self.assertEqual(self.s2s.spec_kwargs["sample_rate"], 16000)
        self.assertEqual(self.s2s.threshold, 1)

    def test_configure(self):
        """Test the configure method of S2SPreProcessor."""
        self.s2s.configure(threshold=2, n_mels=40, hop_length=160)
        self.assertEqual(self.s2s.threshold, 2)
        self.assertEqual(self.s2s.spec_kwargs["n_mels"], 40)
        self.assertEqual(self.s2s.spec_kwargs["hop_length"], 160)

    def test_s2s_shape(self):
        """Test that the S2SPreProcessor returns the correct shape."""
        # Get sample into shape (batch, timestep, features*)
        sample_audio = torch.unsqueeze(self.sample_audio.T, 0)
        sample_audio = torch.tile(sample_audio, (100, 1, 1))

        tensors, targets = self.s2s((sample_audio, torch.Tensor([1] * 100)))
        self.assertEqual(tensors.shape, (100, 60, 20))
        self.assertEqual(targets.shape, (100,))

    def test_s2s_different_batch_sizes(self):
        """Test that the S2SPreProcessor handles different batch sizes correctly."""
        batch_sizes = [10, 20, 256]

        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                sample_audio = torch.unsqueeze(self.sample_audio.T, 0)
                sample_audio = torch.tile(sample_audio, (batch_size, 1, 1))

                tensors, targets = self.s2s(
                    (sample_audio, torch.Tensor([1] * batch_size))
                )
                self.assertEqual(tensors.shape, (batch_size, 60, 20))
                self.assertEqual(targets.shape, (batch_size,))

    def test_s2s_with_kwargs(self):
        """Test that the S2SPreProcessor handles kwargs correctly."""
        sample_audio = torch.unsqueeze(self.sample_audio.T, 0)
        sample_audio = torch.tile(sample_audio, (100, 1, 1))

        kwargs = {"extra_info": "test"}
        tensors, targets, returned_kwargs = self.s2s(
            (sample_audio, torch.Tensor([1] * 100), kwargs)
        )

        self.assertEqual(tensors.shape, (100, 60, 20))
        self.assertEqual(targets.shape, (100,))
        self.assertEqual(returned_kwargs, kwargs)

    def test_s2s_invalid_input(self):
        """Test that the S2SPreProcessor raises an error for invalid input shapes."""
        invalid_audio = torch.rand((10, 5))  # Invalid shape
        self.assertRaises(IndexError, self.s2s, (invalid_audio, torch.Tensor([1] * 10)))
