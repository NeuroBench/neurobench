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
ALL_LANGUAGES = ["en"] #, "es"]
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

    Before the first use, you need to download the english dataset, metadata.json and data splits 
    from the MSWC website and organize them in the neurobench/data/ folder as follows:
    data/
    MSWC/
        metadata.json
        en/
            clips/
            *.csv
            version.txt
    To do so, on https://mlcommons.org/en/multilingual-spoken-words/, follow:
    1.
        Kind -> Language
        Language -> English
        Download "Audio", extract in /data/MSWC, which creates /data/MSWC/en
        Download "Splits", extract in /data/MSWC/en
    2. 
        Kind --> Metadata
        Download in /data/MSWC

    When running for the first time, MSWC will create new data splits csv files based on metadata.json
    and the original csv splits file to select the samples to use for the MSWC FSCIL task.
    """
    def __init__(self, root: Union[str, Path], subset: Optional[str] = None, procedure: Optional[str] = None, 
                 language: Optional[str] = None, incremental: Optional[bool] = False
                 ):
        """ Initialization will create the new base eval splits if needed .
        
        Args:
            root (str): Path of MSWC dataset folder where the Metadata.json file and en/ folders should be.
            subset (str): Return "base" or "evaluation" classes.
            procedure (str): For base subset, return "training", "testing" or "validation" samples.
            languages (str): Language to use for evaluation task.
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

        # if languages is not None and languages != ['en']:
        #     print('Other languages than english are not supported yet.')
        # self.languages = languages if languages is not None else ALL_LANGUAGES

        # If the fscil subset split files don't exist yet, generate them
        # if not os.path.isfile(os.path.join(root, f'{split}.csv')):
        #     generate_mswc_fscil_splits(root, languages)

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


### Generation code ###


def generate_mswc_fscil_splits(root: Union[str, Path], 
                               languages: List[str] = None, 
                               visualize: Optional[bool] = False):
    """
    Generate new MSWC split for a few-shot class-incremental (FSCIL) learning scenario with the following split.
    100 base classes with 500 train, 100 validation and 100 test samples each.
    100 evaluation classes with 200 samples each (to use in a 10 sessions of 10 way set-up with N shots support to train on per class and the rest as a query to evaluate performance).
    The 200 classes are arbitrarily chosen as common voice command words.
    The base ones are then the 100 of these with the most clips (at least 700) per sample and the evaluation ones as the 100 following ones.

    Args
        root (str): Path of MSWC dataset folder where the Metadata.json file and en/ folders should be.
        languages (List[str]): List of languages to use. Not implemented for now, only english will be used.
        visualize (bool): Plots Word Clouds with library wordcloud for a visualization of the FSCIL keywords.

    Returns: base_keywords, evaluation keywords (dictionarries)
    They represent the number of available samples per respective keyword in the original MSWC dataset (although the number is then clipped as detailed above).
    """

    base_keywords, evaluation_keywords = get_command_keywords(root, visualize=visualize)

    if languages is None:
        languages = ['en']

    print(languages)
    if languages  != ['en']:
        print('Other languages than english are not supported yet.')

    base_train_count = dict.fromkeys(base_keywords, 0) #{'train':0, 'val':0, 'test':0})
    base_test_count = dict.fromkeys(base_keywords, 0)
    base_val_count = dict.fromkeys(base_keywords, 0)
    evaluation_count = dict.fromkeys(evaluation_keywords, 0)

    for lang in languages:
        base_train_f = open(os.path.join(root, lang,  f'{lang}_{"base_train"}.csv'), 'w')
        base_val_f = open(os.path.join(root, lang,  f'{lang}_{"base_val"}.csv'), 'w')
        base_test_f = open(os.path.join(root, lang,  f'{lang}_{"base_test"}.csv'), 'w')
        evaluation_f = open(os.path.join(root, lang,  f'{lang}_{"evaluation"}.csv'), 'w')
        writer_base_train = csv.writer(base_train_f)
        writer_base_val = csv.writer(base_val_f)
        writer_base_test = csv.writer(base_test_f)
        writer_evaluation = csv.writer(evaluation_f)
        header = ['LINK', 'WORD', 'VALID', 'SPEAKER', 'GENDER']
        writer_base_train.writerow(header)
        writer_base_val.writerow(header)
        writer_base_test.writerow(header)
        writer_evaluation.writerow(header)

        with open(os.path.join(root, lang,  f'{lang}_splits.csv'), 'r') as f:
            for line in f:
                set, path, word, valid, speaker, gender = line.strip().split(',')
                
                # Skip header
                if set == "SET":
                    continue  

                ### Successively assign samples to train (500), validation (100) and test (100) set
                if word in base_keywords:
                    if base_train_count[word] <500:
                        writer_base_train.writerow([path, word, valid, speaker, gender])
                        base_train_count[word] +=1
                    elif base_val_count[word] <100:
                        writer_base_val.writerow([path, word, valid, speaker, gender])
                        base_val_count[word] +=1
                    elif base_test_count[word] <100:
                        writer_base_test.writerow([path, word, valid, speaker, gender])
                        base_test_count[word] +=1

                elif word in evaluation_keywords:
                    if evaluation_count[word] <200:
                        writer_evaluation.writerow([path, word, valid, speaker, gender])
                        evaluation_count[word] +=1


    base_train_f.close()
    base_val_f.close()
    base_test_f.close()
    evaluation_f.close()

    return base_keywords, evaluation_keywords



def get_command_keywords(root: Union[str, Path], visualize: bool = False):

    with open(os.path.join(root,"metadata.json"), 'r') as f:
        data = json.load(f)
        clips_counts = data['en']['wordcounts']

        mswc_commands = {}

        for keyword in FSCIL_KEYWORDS:
            
            if keyword in clips_counts:
                mswc_commands[keyword] = clips_counts[keyword]

        # Sort keywords based on available clips
        sorted_commands = sorted(mswc_commands.items(), key=lambda x: x[1], reverse=True)

        # Extract the top 200 keywords with the most clips
        pre_train_commands = dict(sorted_commands[:100])
        evaluation_commands = dict(sorted_commands[100:200])

    if visualize:
        from wordcloud import WordCloud

        # Plot word clouds of the base and evaluation words based on their total number of clips in the original MSWC dataset
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(pre_train_commands)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Top 100 Command Keywords with the Most Clips')

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(evaluation_commands)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Evaluation Command Keywords')

        plt.show()  


    return pre_train_commands, evaluation_commands