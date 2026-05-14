from neurobench.processors.preprocessors.mfcc import MFCCPreProcessor
from pathlib import Path
import soundfile
import unittest
import torch


class TestMFCCPreProcessor(unittest.TestCase):
    """Tests MFCCPreProcessor class."""

    def setUp(self):
        """Set up the test data."""
        sample_file = Path(__file__).parent.joinpath("test_data/sample_audio.wav")

        self.sample_audio, self.sampling_rate = soundfile.read(sample_file)
        self.sample_audio = torch.from_numpy(self.sample_audio).float()
        self.init_args = {
            "sample_rate": self.sampling_rate,
            "n_mfcc": 20,
            "dct_type": 2,
            "norm": "ortho",
            "log_mels": False,
            "melkwargs": {
                "n_fft": 256,
                "hop_length": 160,
                "n_mels": 40,
                "center": False,
            },
        }

        # Data is expected in (batch, timesteps, features) format
        self.sample_audio = self.sample_audio.reshape(1, -1, 1)

    def test_mfcc_non_tuple(self):
        """Test that the MFCCPreProcessor raises an error if the input is not a
        tuple."""
        mfcc = MFCCPreProcessor(**self.init_args)
        audio = [1, 2, 3]
        self.assertRaises(TypeError, mfcc, audio)

    def test_mfcc_tuple_wrong_shape(self):
        """Test that the MFCCPreProcessor raises an error if the input tuple has the
        wrong shape."""
        mfcc = MFCCPreProcessor(**self.init_args)
        audio = (self.sample_audio, 2, 3, 4)
        self.assertRaises(ValueError, mfcc, audio)

    def test_mfcc_tuple_correct_shape(self):
        """Test that the MFCCPreProcessor runs correctly with the correct input
        shape."""
        mfcc = MFCCPreProcessor(**self.init_args)
        audio = (self.sample_audio, 2)
        mfcc(audio)
