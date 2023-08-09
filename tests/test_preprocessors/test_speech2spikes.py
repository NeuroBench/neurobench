import unittest
import torch
import torchaudio

from pathlib import Path
from neurobench.preprocessing.speech2spikes import S2SProcessor

class TestSpeech2Spikes(unittest.TestCase):
    def setUp(self):
        sample_file = Path(__file__).parent.joinpath("sample_audio.wav")
        self.sample_audio, self.sampling_rate = torchaudio.load(sample_file)

        # Get sample into shape (batch, timestep, features*)
        self.sample_audio = torch.unsqueeze(self.sample_audio.T, 0)
        self.sample_audio = torch.tile(self.sample_audio, (100, 1, 1))

    def test_s2s(self):
        s2s = S2SProcessor()
        tensors, targets = s2s((self.sample_audio, torch.Tensor([1]*100)))
        self.assertTupleEqual(tensors.shape, (100, 60, 20))
        self.assertTupleEqual(targets.shape, (100,))