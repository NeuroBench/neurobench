"""
"""
import torch
import os
from glob import glob
from torchaudio.datasets import SPEECHCOMMANDS
from .dataset import NeuroBenchDataset

class SpeechCommands(NeuroBenchDataset, SPEECHCOMMANDS):
    """
    Speech commands dataset v0.02 with 35 keywords
    """

    def __init__(self, path, subset:str=None, truncate_or_pad_to_1s=True, ):
        SPEECHCOMMANDS.__init__(self, path, download=True, subset=subset)
        self.truncate_or_pad_to_1s = truncate_or_pad_to_1s

        self.labels = sorted(glob("*/", root_dir=os.path.join(path, "SpeechCommands", "speech_commands_v0.02")))
        # subtract 1 to account for _background_noise_
        self.labels = {key[:-1]: idx-1 for idx, key in enumerate(self.labels)}

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_num =  SPEECHCOMMANDS.__getitem__(self, idx)
        if self.truncate_or_pad_to_1s:
            if waveform.shape[0] > sample_rate:
                waveform = waveform[:sample_rate]
            else:
                temp_waveform = torch.zeros((sample_rate,))
                temp_waveform[:waveform.numel()] = waveform
                waveform = temp_waveform

        waveform = waveform.unsqueeze(-1)
        label = self.label_to_index(label)

        return waveform, label
    
    def label_to_index(self, label):
        return torch.tensor(self.labels[label])

    def __len__(self):
        return SPEECHCOMMANDS.__len__(self)