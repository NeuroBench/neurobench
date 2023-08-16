from typing import Union, Optional, List, Tuple
from pathlib import Path
import os
from torch import Tensor
from torchaudio.datasets.utils import _load_waveform

from torch.utils.data import Dataset

from generation import generate_mswc_fscil_splits

SAMPLE_RATE = 48000
ALL_LANGUAGES = ["en"] #, "es"]
FOLDER_AUDIO = "clips"

PRE_TRAINING_TAGS = []
EVALUATION_TAGS = []

def _load_list(root: Union[str, Path], languages: List[str], split: str) -> List[Tuple[str, str, bool, str, str]]:
    walker = []

    for lang in languages:
        with open(os.path.join(root, lang, f'{lang}_{split}.csv'), 'r') as f:
            for line in f:
                path, word, valid, speaker, gender = line.strip().split(',')
                
                # Skip header
                if word == "WORD":
                    continue        

                walker.append((path, word, bool(valid), speaker, gender, lang))

    return walker


class MSWC(Dataset):
    def __init__(self, root: Union[str, Path], subset: Optional[str] = None, procedure: Optional[str] = None, languages: Optional[List[str]] = None):
        self.root = root

        if subset == 'base':
            self.subset = 'base'
            if procedure == "training":
                self.procedure = "train"
            elif procedure == "validation":
                self.procedure = "val"
            elif procedure == "testing":
                self.procedure = "test"
            else:
                raise ValueError("procedure must be one of \"training\", \"validation\", or \"testing\"")

            split = self.subset +'_' + self.procedure
        elif subset == 'evaluation':
            self.subset = 'evaluation'
            self.procedure = None
            split = self.subset

        else:
            raise ValueError("subset must be one of \"base\" or \"evaluation\"")

        if languages is not None and languages != ['en']:
            print('Other languages than english are not supported yet.')
        # self.languages = languages if languages is not None else ALL_LANGUAGES

        # If the fscil subset split files don't exist anymore, generate them
        if not os.path.isfile(os.path.join(root, 'en', f'{"en"}_{split}.csv')):
            generate_mswc_fscil_splits(root, languages)

        self._walker = _load_list(root, self.languages, split)

    def __getitem__(self, index: int) -> Tuple[Tensor, str, bool, str, str, str]:
        item = self._walker[index]

        dirname = os.path.join(self.root, item[5], FOLDER_AUDIO)
        waveform = _load_waveform(dirname, item[0], SAMPLE_RATE)

        return (waveform, *item[1:])
    