import torch
import os
from glob import glob
from torchaudio.datasets import SPEECHCOMMANDS
from .dataset import NeuroBenchDataset


class SpeechCommands(NeuroBenchDataset, SPEECHCOMMANDS):
    """
    Speech commands dataset v0.02 with 35 keywords.

    Wraps the torchaudio SPEECHCOMMANDS dataset.

    """

    def __init__(
        self,
        path,
        subset: str = None,
        truncate_or_pad_to_1s=True,
    ):
        """
        Initializes the SpeechCommands dataset.

        Args:
            path (str): path to the root directory of the dataset
            subset (str, optional): one of "training", "validation", or "testing". Defaults to None.
            truncate_or_pad_to_1s (bool, optional): whether to truncate or pad samples to 1s. Defaults to True.

        """
        os.makedirs(path, exist_ok=True)
        SPEECHCOMMANDS.__init__(self, path, download=True, subset=subset)
        self.truncate_or_pad_to_1s = truncate_or_pad_to_1s

        # convert labels to indices
        self.labels = sorted(
            glob(
                "*/",
                root_dir=os.path.join(path, "SpeechCommands", "speech_commands_v0.02"),
            )
        )
        # subtract 1 to account for _background_noise_
        self.labels = {key[:-1]: idx - 1 for idx, key in enumerate(self.labels)}

    def __getitem__(self, idx):
        """
        Getter method for dataset.

        Args:
            idx (int): index of sample to return
        Returns:
            waveform (torch.Tensor): waveform of audio sample
            label (torch.Tensor): label index of audio sample

        """
        (
            waveform,
            sample_rate,
            label,
            speaker_id,
            utterance_num,
        ) = SPEECHCOMMANDS.__getitem__(self, idx)
        if self.truncate_or_pad_to_1s:
            if waveform.shape[0] > sample_rate:
                waveform = waveform[:sample_rate]
            else:
                temp_waveform = torch.zeros((sample_rate,))
                temp_waveform[: waveform.numel()] = waveform
                waveform = temp_waveform

        waveform = waveform.unsqueeze(-1)
        label = self.label_to_index(label)

        return waveform, label

    def label_to_index(self, label):
        """
        Converts a label to an index.

        Args:
            label (str): label of audio sample
        Returns:
            torch.Tensor: index of label

        """
        return torch.tensor(self.labels[label])

    def __len__(self):
        """
        Returns number of samples in dataset.

        Returns:
            int: number of samples in dataset

        """
        return SPEECHCOMMANDS.__len__(self)
