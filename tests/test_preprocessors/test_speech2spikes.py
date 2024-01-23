from pathlib import Path

import torch
import torchaudio

from neurobench.preprocessing.speech2spikes import S2SPreProcessor

def test_s2s():
    sample_file = Path(__file__).parent.joinpath("sample_audio.wav")
    sample_audio, sampling_rate = torchaudio.load(sample_file)

    # Get sample into shape (batch, timestep, features*)
    sample_audio = torch.unsqueeze(sample_audio.T, 0)
    sample_audio = torch.tile(sample_audio, (100, 1, 1))

    s2s = S2SPreProcessor()
    tensors, targets = s2s((sample_audio, torch.Tensor([1]*100)))
    assert tensors.shape == (100, 60, 20)
    assert targets.shape == (100,)