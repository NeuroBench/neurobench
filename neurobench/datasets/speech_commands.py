"""
"""
import torch
from torchaudio.datasets import SPEECHCOMMANDS
from .dataset import NeuroBenchDataset

class SpeechCommands(NeuroBenchDataset, SPEECHCOMMANDS):
    """
    Speech commands dataset v0.02 with 35 keywords
    """

    def __init__(self, subset:str=None, truncate_or_pad_to_1s=True, ):
        SPEECHCOMMANDS.__init__(self, "./", download=True, subset=subset)
        self.truncate_or_pad_to_1s = truncate_or_pad_to_1s

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_num =  SPEECHCOMMANDS.__getitem__(self, idx)
        if self.truncate_or_pad_to_1s:
            if waveform.shape[0] > sample_rate:
                waveform = waveform[:sample_rate]
            else:
                temp_waveform = torch.zeros((sample_rate,))
                temp_waveform[:waveform.numel()] = waveform
                waveform = temp_waveform
        
        return waveform, label
    
    def __len__(self):
        return SPEECHCOMMANDS.__len__(self)


    