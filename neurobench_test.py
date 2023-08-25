import torch

# from speech2spikes import S2S

import torchvision
from torch.utils.data import DataLoader

from torch_mate.data.utils import FewShot
from torch_mate.utils import get_device

from neurobench.data.datasets.MSWC import MSWC
from neurobench.examples.few_shot_learning.utils import train_using_MAML
from neurobench.models import M5
from neurobench.utils import Dict2Class

ROOT = "//scratch/p306982/data/fscil/mswc/"

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
        "query_shots": 100,
    },
    "continual_learning": {
        "max_classes_to_learn": 200
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

eval_dataset = MSWC(ROOT, subset='evaluation')

fscil_set = FewShot(eval_dataset, 10, 5, 100, None, (100, 100), True, True, None, 200, torch.nn.Identity(), None)

eval_data_loader = DataLoader(fscil_set, 1, num_workers=8)

for session, (X, y) in enumerate(fscil_set):
    print("Session: {}".format(session))
    X_train, X_test = X
    y_train, y_test = y
    
    # TRAIN model using train_data
    # model = train(model, (X_train, y_train))


    ### TEST
    # y_pred_test = model(X_test)

    # print('test')
    ## run benchmark ##
    # session_results = benchmark.run()

    # print(session_results)

