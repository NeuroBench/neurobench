import unittest
from pathlib import Path
from neurobench.preprocessing.mfcc import MFCCProcessor
import torch
import torchaudio
import numpy as np


class TestMFCCProcessor(unittest.TestCase):
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

    def test_mfcc_non_tuple(self):
        mfcc = MFCCProcessor(**self.init_args)
        audio = [1, 2, 3]
        with self.assertRaises(TypeError):
            mfcc(audio)

    def test_mfcc_tuple_wrong_shape(self):
        mfcc = MFCCProcessor(**self.init_args)
        audio = (self.sample_audio, 2, 3)
        with self.assertRaises(ValueError):
            mfcc(audio)

    def test_mfcc_tuple_correct_shape(self):
        mfcc = MFCCProcessor(**self.init_args)
        audio = (self.sample_audio, 2)
        mfcc(audio)

    def test_mfcc_audio(self):
        mfcc = MFCCProcessor(**self.init_args)
        from librosa.feature.spectral import mfcc as librosa_mfcc

        librosa_mfcc_params = ({"n_mfcc": self, "n_fft": 512, "hop_length": 512},)

        mfcc_reference = librosa_mfcc(
            y=self.sample_audio.numpy(),
            sr=self.init_args["sample_rate"],
            n_mfcc=self.init_args["n_mfcc"],
            norm=self.init_args["norm"],
            dct_type=self.init_args["dct_type"],
            **self.init_args["melkwargs"]
        )

        result = mfcc((self.sample_audio, 0))

        self.assertLessEqual(np.max(mfcc_reference - result[0].numpy()), 1e-3)
