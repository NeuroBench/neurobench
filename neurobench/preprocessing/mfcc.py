from . import NeuroBenchProcessor
from torchaudio.transforms import MFCC
import torch


class MFCCProcessor(NeuroBenchProcessor):
    """ Does MFCC computation on dataset using torchaudio.transforms.MFCC.
    Call expects loaded .wav data and targets as a tuple (data, targets).
    Expects sample_rate to be the same for all samples in data.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        melkwargs: dict = None,
    ):
        super(NeuroBenchProcessor).__init__()
        """
        Args:
            sample_rate (int, optional): Sample rate of the audio signal. (Default: 16000)
            n_mfcc (int, optional): Number of MFCC coefficients to retain. (Default: 40)
            dct_type (int, optional): Type of DCT (discrete cosine transform) to use. (Default: 2)
            norm (str, optional): Norm to use. (Default: "ortho")
            log_mels (bool, optional): Whether to use log-mel spectrograms instead of db-scaled. (Default: False)
            melkwargs (dict or None, optional): Arguments for MelSpectrogram. (Default: None)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = 2
        self.norm = norm
        self.log_mels = log_mels
        self.melkwargs = melkwargs

        self.mfcc = MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            log_mels=self.log_mels,
            melkwargs=self.melkwargs,
        )

    def __call__(self, dataset):
        """ Executes the MFCC computation on the dataset.

        Args:
            dataset (tuple): A tuple of (data, targets).

        Returns:
            results: mfcc applied on data
            targets: targets from dataset
        """
        self.dataset_validity_check(dataset)

        data, targets = dataset
        if isinstance(data, list):
            data = torch.vstack(data)

        self.results = self.mfcc(data)

        return self.results, targets

    @staticmethod
    def dataset_validity_check(dataset):
        """ Checks if dataset is a tuple with length two.
        """
        if not isinstance(dataset, tuple):
            raise TypeError("Expected dataset to be tuple")

        if not len(dataset) == 2:
            raise ValueError("Dataset tuple should have values as (data, targets)")
