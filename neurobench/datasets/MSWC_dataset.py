from typing import Union, Optional, List, Tuple
from pathlib import Path
import os
from torch import Tensor
from torchaudio.datasets.utils import _load_waveform
import torch
from torch.utils.data import Dataset
from .utils import download_url
from urllib.error import URLError
import tarfile
from tqdm import tqdm

FSCIL_KEYWORDS = [
    "surrounding",
    "appearance",
    "collection",
    "subsequently",
    "experience",
    "previously",
    "professional",
    "immediately",
    "independent",
    "successful",
    "traditional",
    "development",
    "international",
    "established",
    "information",
    "instruments",
    "construction",
    "performance",
    "university",
    "significant",
    "nombreuses",
    "l’amendement",
    "nationale",
    "défavorable",
    "toutefois",
    "professeur",
    "française",
    "commission",
    "actuellement",
    "politique",
    "territoire",
    "aujourd’hui",
    "gouvernement",
    "rapidement",
    "notamment",
    "troisième",
    "plusieurs",
    "maintenant",
    "personnes",
    "cinquante",
    "tanmateix",
    "superior",
    "catalunya",
    "finalment",
    "treballar",
    "qualsevol",
    "principalment",
    "rectangular",
    "important",
    "mitjançant",
    "president",
    "finestres",
    "caràcter",
    "diferents",
    "programari",
    "obertures",
    "posteriorment",
    "barcelona",
    "importants",
    "informació",
    "allerdings",
    "arbeitete",
    "deutschland",
    "schließlich",
    "verheiratet",
    "vielleicht",
    "verschiedene",
    "bezeichnet",
    "gleichzeitig",
    "parlament",
    "unternehmen",
    "entwicklung",
    "europäischen",
    "zahlreiche",
    "geschichte",
    "verwendet",
    "anschließend",
    "hauptstadt",
    "errichtet",
    "eigentlich",
    "amahirwe",
    "umutekano",
    "gukomeza",
    "amafaranga",
    "abanyarwanda",
    "urubyiruko",
    "n’abandi",
    "ashobora",
    "umuryango",
    "abayobozi",
    "w’umupira",
    "perezida",
    "umukinnyi",
    "gitaramo",
    "w’amaguru",
    "abaturage",
    "amashusho",
    "minisitiri",
    "indirimbo",
    "bitandukanye",
    "میخواستم",
    "امیدوارم",
    "تلویزیون",
    "آپارتمان",
    "بینالمللی",
    "بنابراین",
    "اسفندیار",
    "پرسپولیس",
    "بفرمایید",
    "میتوانند",
    "demandis",
    "respondis",
    "sinjorino",
    "rigardis",
    "fernando",
    "ankoraŭ",
    "beatrico",
    "troviĝas",
    "ekzistas",
    "esperanto",
    "enquanto",
    "correndo",
    "olhando",
    "dinheiro",
    "vestindo",
    "segurando",
    "crianças",
    "mulheres",
    "trabalho",
    "cachorro",
    "bakarrik",
    "ezagutzen",
    "hizkuntza",
    "horregatik",
    "bitartean",
    "erabiltzen",
    "horrelako",
    "gertatzen",
    "eskatzen",
    "batzuetan",
    "podlasiak",
    "zupełnie",
    "odpowiedział",
    "dziwożona",
    "jednoręki",
    "widocznie",
    "powiedział",
    "spojrzał",
    "człowiek",
    "wszystkich",
    "gobeithio",
    "newyddion",
    "wybodaeth",
    "ddigwyddodd",
    "eisteddfod",
    "genedlaethol",
    "gwasanaeth",
    "gwahaniaeth",
    "ffrindiau",
    "gerddoriaeth",
    "gezien",
    "iedereen",
    "tussen",
    "kinderen",
    "tijdens",
    "worden",
    "hoeveel",
    "kunnen",
    "nieuwe",
    "gemaakt",
    "приверженность",
    "действительно",
    "сотрудничества",
    "необходимость",
    "деятельности",
    "правительства",
    "поблагодарить",
    "представителю",
    "международного",
    "ответственность",
    "comunicación",
    "arquitectura",
    "completamente",
    "continuación",
    "departamento",
    "internacional",
    "especialmente",
    "posteriormente",
    "investigación",
    "generalmente",
    "particolare",
    "territorio",
    "campionato",
    "importante",
    "capoluogo",
    "nonostante",
    "attraverso",
    "posizione",
    "soprattutto",
    "produzione",
]

SAMPLE_RATE = 48000
FOLDER_AUDIO = "clips"


def _load_list(split_path: Union[str, Path]) -> List[Tuple[str, str, bool, str, str]]:
    walker = []

    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            path, word, lang = line.strip().split(",")

            # Skip header
            if word == "WORD":
                continue

            index = FSCIL_KEYWORDS.index(word)

            walker.append((path, index, lang))

    return walker


def get_mswc_item(item, dirname, return_path):
    waveform = _load_waveform(dirname, item[0], SAMPLE_RATE)

    if waveform.size()[1] != SAMPLE_RATE:
        full_size = torch.zeros((1, SAMPLE_RATE))
        full_size[:, : waveform.size()[1]] = waveform
        waveform = full_size

    # Data is expected to be (timesteps, features)
    waveform = waveform.permute(1, 0)

    if return_path:
        return (waveform, item[1], dirname, item[0])
    else:
        return (waveform, item[1])


class MSWC(Dataset):
    """
    Subset version (https://huggingface.co/datasets/NeuroBench/mswc_fscil_subset)
    of the original MSWC dataset (https://mlcommons.org/en/multilingual-spoken-words/)
    for a few-shot class-incremental learning (FSCIL) task consisting of 200 voice commands keywords:

    - 100 base classes available for pre-training with:
        - 500 train samples
        - 100 validation samples
        - 100 test samples
    - 100 evaluation classes to do class-incremental learning on with 200 samples each.

    The subset of data used for this task, as well as the supporting files for base class and incremental
    splits, is hosted on Huggingface at the first link above.

    The data is given in 48kHz opus format. Converted 16kHz wav files are available to download at the link above.
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: Optional[str] = None,
        procedure: Optional[str] = None,
        language: Optional[str] = None,
        incremental: Optional[bool] = False,
        download=True,
    ):
        """
        Initialization will create the new base eval splits if needed .

        Args:
            root (str): Path of data root folder where is or will be the MSWC/ folder containing the dataset.
            subset (str): Return "base" or "evaluation" classes.
            procedure (str): For base subset, return "training", "testing" or "validation" samples.
            language (str): Language to use for evaluation task.
            download (bool): If True, downloads the dataset from the internet and puts it in root
                                 directory. If dataset is already downloaded, it will not be downloaded again.

        """
        self.root = root
        self.dataset_folder = os.path.join(root, "MSWC")

        if subset == "base":
            self.subset = "base"
            if procedure == "training":
                self.procedure = "train"
            elif procedure == "validation":
                self.procedure = "val"
            elif procedure == "testing":
                self.procedure = "test"
            else:
                raise ValueError(
                    'procedure must be one of "training", "validation", or "testing"'
                )

            split = self.subset + "_" + self.procedure
            split_path = os.path.join(self.dataset_folder, f"{split}.csv")
            if incremental:
                self.return_path = True
            else:
                self.return_path = False

        elif subset == "evaluation":
            self.subset = "evaluation"
            self.procedure = None
            split = self.subset
            split_path = os.path.join(self.dataset_folder, language, f"{split}.csv")
            self.return_path = True

        else:
            raise ValueError('subset must be one of "base" or "evaluation"')

        self.url = "https://huggingface.co/datasets/NeuroBench/mswc_fscil_subset/resolve/main/mswc_fscil.tar.gz"

        if download and not os.path.exists(self.dataset_folder):
            print("downloading ....")
            self.download()

        self._walker = _load_list(split_path)

    def download(self):
        """Download the MSWC FSCIL data if it doesn't exist already."""

        if os.path.exists(self.dataset_folder):
            print("The dataset already exists!")
            return

        os.makedirs(os.path.dirname(self.dataset_folder), exist_ok=True)

        # download file
        file_path = f"{os.path.dirname(self.dataset_folder)}/mswc_fscil.tar.gz"
        try:
            print(f"Downloading {self.url}")
            download_url(self.url, file_path)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print("Unzipping file...")

            with tarfile.open(file_path, "r:gz") as tar:
                total_files = len(tar.getmembers())

                # Set up the tqdm progress bar
                with tqdm(
                    total=total_files, unit="file", desc="Extracting files"
                ) as progress_bar:
                    for member in tar.getmembers():
                        tar.extract(member, path=self.root)
                        progress_bar.update(1)
            print()

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Getter method to get waveform samples.

        Args:
            idx (int): Index of the sample.

        Returns:
            sample (tensor): Individual waveform sample, padded to always match dimension (48000, 1).
            target (int): Corresponding keyword index based on FSCIL_KEYWORDS order (by decreasing number of samples in original dataset).

        """
        item = self._walker[index]

        dirname = os.path.join(self.dataset_folder, item[2], FOLDER_AUDIO)

        return get_mswc_item(item, dirname, self.return_path)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self._walker)


class MSWC_query(Dataset):
    """Simple Dataset object created for incremental queries."""

    def __init__(self, walker):
        """
        Initialization of the dataset.

        Args:
            walker (list): List of tuples with data (filename, class_index, dirname)

        """

        self._walker = walker

    def __getitem__(self, index: int):
        """
        Getter method to get waveform samples.

        Args:
            idx (int): Index of the sample.

        Returns:
            sample (tensor): Individual waveform sample, padded to always match dimension (1, 48000).
            target (int): Corresponding keyword index based on FSCIL_KEYWORDS order (by decreasing number of samples in original dataset).

        """
        item = self._walker[index]

        return get_mswc_item(item, item[2], False)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self._walker)
