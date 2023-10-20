from typing import Union, Optional, List, Tuple
from pathlib import Path
import os
from torch import Tensor
from torchaudio.datasets.utils import _load_waveform
import torch

import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

FSCIL_KEYWORDS = ['surrounding',
 'appearance',
 'collection',
 'subsequently',
 'experience',
 'previously',
 'professional',
 'immediately',
 'independent',
 'successful',
 'traditional',
 'development',
 'international',
 'established',
 'information',
 'instruments',
 'construction',
 'performance',
 'university',
 'significant',
 'nombreuses',
 'l’amendement',
 'nationale',
 'défavorable',
 'toutefois',
 'professeur',
 'française',
 'commission',
 'actuellement',
 'politique',
 'territoire',
 'aujourd’hui',
 'gouvernement',
 'rapidement',
 'notamment',
 'troisième',
 'plusieurs',
 'maintenant',
 'personnes',
 'cinquante',
 'tanmateix',
 'superior',
 'catalunya',
 'finalment',
 'treballar',
 'qualsevol',
 'principalment',
 'rectangular',
 'important',
 'mitjançant',
 'president',
 'finestres',
 'caràcter',
 'diferents',
 'programari',
 'obertures',
 'posteriorment',
 'barcelona',
 'importants',
 'informació',
 'allerdings',
 'arbeitete',
 'deutschland',
 'schließlich',
 'verheiratet',
 'vielleicht',
 'verschiedene',
 'bezeichnet',
 'gleichzeitig',
 'parlament',
 'unternehmen',
 'entwicklung',
 'europäischen',
 'zahlreiche',
 'geschichte',
 'verwendet',
 'anschließend',
 'hauptstadt',
 'errichtet',
 'eigentlich',
 'amahirwe',
 'umutekano',
 'gukomeza',
 'amafaranga',
 'abanyarwanda',
 'urubyiruko',
 'n’abandi',
 'ashobora',
 'umuryango',
 'abayobozi',
 'w’umupira',
 'perezida',
 'umukinnyi',
 'gitaramo',
 'w’amaguru',
 'abaturage',
 'amashusho',
 'minisitiri',
 'indirimbo',
 'bitandukanye',
 'میخواستم',
 'امیدوارم',
 'تلویزیون',
 'آپارتمان',
 'بینالمللی',
 'بنابراین',
 'اسفندیار',
 'پرسپولیس',
 'بفرمایید',
 'میتوانند',
 'demandis',
 'respondis',
 'sinjorino',
 'rigardis',
 'fernando',
 'ankoraŭ',
 'beatrico',
 'troviĝas',
 'ekzistas',
 'esperanto',
 'enquanto',
 'correndo',
 'olhando',
 'dinheiro',
 'vestindo',
 'segurando',
 'crianças',
 'mulheres',
 'trabalho',
 'cachorro',
 'bakarrik',
 'ezagutzen',
 'hizkuntza',
 'horregatik',
 'bitartean',
 'erabiltzen',
 'horrelako',
 'gertatzen',
 'eskatzen',
 'batzuetan',
 'podlasiak',
 'zupełnie',
 'odpowiedział',
 'dziwożona',
 'jednoręki',
 'widocznie',
 'powiedział',
 'spojrzał',
 'człowiek',
 'wszystkich',
 'gobeithio',
 'newyddion',
 'wybodaeth',
 'ddigwyddodd',
 'eisteddfod',
 'genedlaethol',
 'gwasanaeth',
 'gwahaniaeth',
 'ffrindiau',
 'gerddoriaeth',
 'gezien',
 'iedereen',
 'tussen',
 'kinderen',
 'tijdens',
 'worden',
 'hoeveel',
 'kunnen',
 'nieuwe',
 'gemaakt',
 'приверженность',
 'действительно',
 'сотрудничества',
 'необходимость',
 'деятельности',
 'правительства',
 'поблагодарить',
 'представителю',
 'международного',
 'ответственность',
 'comunicación',
 'arquitectura',
 'completamente',
 'continuación',
 'departamento',
 'internacional',
 'especialmente',
 'posteriormente',
 'investigación',
 'generalmente',
 'particolare',
 'territorio',
 'campionato',
 'importante',
 'capoluogo',
 'nonostante',
 'attraverso',
 'posizione',
 'soprattutto',
 'produzione']

SAMPLE_RATE = 48000
FOLDER_AUDIO = "clips"

def _load_list(split_path: Union[str, Path]) -> List[Tuple[str, str, bool, str, str]]:
    walker = []

    with open(split_path, 'r') as f:
        for line in f:
            path, word, lang = line.strip().split(',')
            
            # Skip header
            if word == "WORD":
                continue        
            
            index = FSCIL_KEYWORDS.index(word)

            walker.append((path, index, lang))

    return walker


class MSWC(Dataset):
    """ 
    Subset version of the original MSWC dataset (https://mlcommons.org/en/multilingual-spoken-words/)
    for a few-shot class-incremental learning (FSCIL) task consisting of 200 voice commands keywords:
    - 100 base classes available for pre-training with:
        - 500 train samples
        - 100 validation samples
        - 100 test samples
    - 100 evaluation classes to do class-incremental learning on with 200 samples each.

    The subset of data used for this task, as well as the supporting files for base class and incremental 
    splits, will be hosted and available shortly. If you are interested in using this dataset beforehand, 
    please email jyik@g.harvard.edu for dataset download.
    
    The data should be organized as follows:
    data/
    MSWC/
        base_[test,train,val].csv
        language/ (for all languages in general or FSCIL dataset)
            clips/
            *.csv
    """
    def __init__(self, root: Union[str, Path], subset: Optional[str] = None, procedure: Optional[str] = None, 
                 language: Optional[str] = None, incremental: Optional[bool] = False
                 ):
        """ Initialization will create the new base eval splits if needed .
        
        Args:
            root (str): Path of MSWC dataset folder where the Metadata.json file and en/ folders should be.
            subset (str): Return "base" or "evaluation" classes.
            procedure (str): For base subset, return "training", "testing" or "validation" samples.
            language (str): Language to use for evaluation task.
        """
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
            split_path = os.path.join(root, f'{split}.csv')
            if incremental:
                self.return_path = True
            else:
                self.return_path = False
                

        elif subset == 'evaluation':
            self.subset = 'evaluation'
            self.procedure = None
            split = self.subset
            split_path = os.path.join(root, language, f'{split}.csv')
            self.return_path = True

        else:
            raise ValueError("subset must be one of \"base\" or \"evaluation\"")

        self._walker = _load_list(split_path)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """ Getter method to get waveform samples.

        Args:
            idx (int): Index of the sample.

        Returns:
            sample (tensor): Individual waveform sample, padded to always match dimension (1, 48000).
            target (int): Corresponding keyword index based on FSCIL_KEYWORDS order (by decreasing number of samples in original dataset).
        """
        item = self._walker[index]

        dirname = os.path.join(self.root, item[2], FOLDER_AUDIO)
        waveform = _load_waveform(dirname, item[0], SAMPLE_RATE)

        if waveform.size()[1] != SAMPLE_RATE:
            full_size = torch.zeros((1,SAMPLE_RATE))
            full_size[:, :waveform.size()[1]] = waveform
            waveform = full_size

        if self.return_path:
            return (waveform, item[1], dirname, item[0])
        else:
            return (waveform, item[1])

    def __len__(self):
        """ Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return(len(self._walker))



class MSWC_query(Dataset):
    """
    Simple Dataset object created for incremental queries
    """

    def __init__(self, walker):

        self._walker = walker

    def __getitem__(self, index: int):
        """ Getter method to get waveform samples.

        Args:
            idx (int): Index of the sample.

        Returns:
            sample (tensor): Individual waveform sample, padded to always match dimension (1, 48000).
            target (int): Corresponding keyword index based on FSCIL_KEYWORDS order (by decreasing number of samples in original dataset).
        """
        item = self._walker[index]

        dirname = item[0]
        waveform = _load_waveform(dirname, item[1], SAMPLE_RATE)

        if waveform.size()[1] != SAMPLE_RATE:
            full_size = torch.zeros((1,SAMPLE_RATE))
            full_size[:, :waveform.size()[1]] = waveform
            waveform = full_size


        return (waveform, item[2])

    def __len__(self):
        """ Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return(len(self._walker))
