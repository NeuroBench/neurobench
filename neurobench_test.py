import torch

# from speech2spikes import S2S

import torchvision
from torch.utils.data import DataLoader, ConcatDataset
import torchaudio

from torch_mate.data.utils import FewShot, IncrementalFewShot
from torch_mate.utils import get_device

from neurobench.data.datasets.MSWC import MSWC
from neurobench.examples.few_shot_learning.utils import train_using_MAML
from neurobench.models import M5
from neurobench.utils import Dict2Class

from neurobench.benchmarks import Benchmark


import timeit
ROOT = "//scratch/p306982/data/fscil/mswc/"


train = lambda net, data: net

model = M5(n_input=48, n_output=40)
eval_set = MSWC(root=ROOT, subset="evaluation")
# transform = torchaudio.transforms.MFCC(sample_rate: int = 16000, n_mfcc: int = 40, dct_type: int = 2, norm: str = 'ortho', log_mels: bool = False, melkwargs: Optional[dict] = None)

few_shot_dataloader = FewShot(eval_set, 10, 5, 100,
                            incremental=True,
                            cumulative=True,
                            support_query_split=(100,100),
                            samples_per_class=200,
                            transform=torch.nn.Identity())

# few_shot_dataloader = DataLoader(few_shot_dataloader, batch_size=1, num_workers=8)


base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")

benchmark = Benchmark(model, metric_list=[[],["classification_accuracy"]], dataloader=None, preprocessors=[torch.nn.Identity()], postprocessors=[torch.nn.Identity()])


for session, (X, y) in enumerate(few_shot_dataloader):
    print(f"Session: {session}")

    X_train, X_test = X
    y_train, y_test = y

    # X_train = X_train.squeeze()
    # y_train = y_train.squeeze()
    # X_test = X_test.squeeze()
    # y_test = y_test.squeeze()


    # model = train(model, (X_train[0], y_train[0]))

    # ### Testing phase ###
    # session_query_set = torch.utils.data.TensorDataset(X_test,y_test)
    # full_session_test_set = ConcatDataset([base_test_set, session_query_set])
    # full_session_test_loader = DataLoader(full_session_test_set, batch_size=256, num_workers=8)

    # session_results = benchmark.run(data = full_session_test_loader)

    # print(session_results)
