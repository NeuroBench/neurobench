import torchvision
from torch.utils.data import DataLoader

from torch_mate.data.utils import FewShot
from torch_mate.utils import get_device

from neurobench.examples.maml.utils import train_using_maml
from neurobench.models import OmniglotCNN
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
    "model": {
        "name": "OmniglotCNN",
        "cfg": {
            "input_channels": 1,
            "hidden_channels": 64
        },
    },
    "optimizer": {"name": "Adam", "cfg": {"lr": 0.001, "betas": (0.9, 0.999)}},
    "seed": 4223747124,
    "task": {
        "name": "Omniglot",
    },
}

cfg = Dict2Class(cfg)

model = OmniglotCNN(cfg.model.cfg.input_channels,
                    cfg.model.cfg.hidden_channels,
                    cfg.model.cfg.hidden_channels,
                    cfg.meta_learning.ways)

train_dataset = torchvision.datasets.Omniglot('../../data/datasets/data', background=True, download=True)
test_dataset = torchvision.datasets.Omniglot('../../data/datasets/data', background=False, download=True)

few_shot_args = (cfg.meta_learning.shots, cfg.meta_learning.query_shots, None, None, None)

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

train_using_maml(
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
