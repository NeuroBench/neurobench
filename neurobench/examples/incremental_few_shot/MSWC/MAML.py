import torchvision
from torch.utils.data import DataLoader

from torch_mate.data.utils import FewShot
from torch_mate.utils import get_device

from neurobench.datasets import MSWC
from neurobench.examples.few_shot_learning.utils import train_using_MAML
from neurobench.models import M5
from neurobench.utils import Dict2Class

cfg = {
    "criterion": {"name": "CrossEntropyLoss"},
    "meta_learning": {
        "fast_lr": 0.4,
        "adaptation_steps": 1,
        "test_adaptation_steps": 1,
        "meta_batch_size": 32,
        "num_iterations": 30000,
        "name": "MAML",
        "first_order": False,
        "ways": 5,
        "shots": 1,
        "query_shots": 1,
    },
    "continual_learning": {
        "max_classes_to_learn": 100
    },
    "model": {
        "name": "M5",
        "cfg": {
            "stride": 16,
            "n_channel": 32
        },
    },
    "optimizer": {"name": "Adam", "cfg": {"lr": 0.001, "betas": (0.9, 0.999)}},
    "seed": 4223747124,
    "task": {
        "name": "MWSC",
        "cfg": {
            "representation": {
                "name": "MFCC",
                "cfg": {
                    "center": True,
                    "hop_length": 160,
                    "n_fft": 400,
                    "n_mels": 96,
                    "n_mfcc": 48

                }
            }
        }
    },
}

cfg = Dict2Class(cfg)

model = M5(n_input=cfg.task.cfg.representation.cfg.n_mfcc,
           stride=cfg.model.cfg.stride,
           n_channel=cfg.model.cfg.n_channel,
           n_output=cfg.continual_learning.max_classes_to_learn)

train_dataset = torchvision.datasets.MSWC('../../data/datasets/data', subset='train')
test_dataset = torchvision.datasets.MSWC('../../data/datasets/data', subset='test')

few_shot_args = (cfg.meta_learning.shots, cfg.meta_learning.query_shots, True, None, None, None)

train_data_loader = DataLoader(FewShot(
    train_dataset,
    cfg.meta_learning.train_ways,
    *few_shot_args
), batch_size=cfg.meta_learning.meta_batch_size,
num_workers=8)

test_data_loader = DataLoader(FewShot(
    test_dataset,
    cfg.meta_learning.ways,
    *few_shot_args
), batch_size=cfg.meta_learning.meta_batch_size,
num_workers=8)

device = get_device()

train_using_MAML(
    model,
    train_data_loader,
    test_data_loader,
    device,
    "./models",
    cfg,
    log=print,
    test_every=20,
    save_every=1000
)
