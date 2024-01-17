import unittest
from pathlib import Path
from neurobench.preprocessing.mfcc import MFCCPreProcessor
import torch
import torchaudio
import numpy as np


class TestMFCCPreProcessor(unittest.TestCase):
    def setUp(self):
        sample_file = Path(__file__).parent.joinpath("sample_audio.wav")

        self.sample_audio, self.sampling_rate = torchaudio.load(sample_file)
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
        self.sample_audio = self.sample_audio.permute(0, 1)

    def test_mfcc_non_tuple(self):
        mfcc = MFCCPreProcessor(**self.init_args)
        audio = [1, 2, 3]
        with self.assertRaises(TypeError):
            mfcc(audio)

    def test_mfcc_tuple_wrong_shape(self):
        mfcc = MFCCPreProcessor(**self.init_args)
        audio = (self.sample_audio, 2, 3, 4)
        with self.assertRaises(ValueError):
            mfcc(audio)

    def test_mfcc_tuple_correct_shape(self):
        mfcc = MFCCPreProcessor(**self.init_args)
        audio = (self.sample_audio, 2)
        mfcc(audio)
