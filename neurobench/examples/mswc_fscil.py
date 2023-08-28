#
# NOTE: This task is still under development.
#

import torch

from torch.utils.data import DataLoader, ConcatDataset
import torchaudio

from torch_mate.data.utils import FewShot

from neurobench.datasets import MSWC
from neurobench.examples.model_data.M5 import M5

from neurobench.benchmarks import Benchmark

ROOT = "data/MSWC/"
NUM_WORKERS = 1

dummy_train = lambda net, data: net


model = M5(n_input=1, n_output=200)

### Pre-training phase ###
base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
# @TODO Add your own pre-training on the base_train and base_val classes here 
model = dummy_train(model, base_train_set)

del base_train_set


### Evaluation phase ###

# Get Datasets: evaluation + all test samples from base classes to test forgetting
eval_set = MSWC(root=ROOT, subset="evaluation")
base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")

# FewShot Dataloader used in incremental mode to generate class-incremental sessions
few_shot_dataloader = FewShot(eval_set, n_way=10, k_shot=5, query_shots=100,
                            incremental=True,
                            cumulative=True,
                            support_query_split=(100,100),
                            samples_per_class=200)




# Define an arbitrary resampling as an example of pre-processor to feed to the Benchmark object
new_sample_rate = 8000
resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=new_sample_rate)
pre_proc_resample = lambda x: (resample(x[0]), x[1])

# Define benchmark object
benchmark = Benchmark(model, metric_list=[[],["classification_accuracy"]], dataloader=None, preprocessors=[pre_proc_resample], postprocessors=[torch.nn.Identity()])
all_results = []

# Iteration over incremental sessions
for session, (X, y) in enumerate(few_shot_dataloader):
    print(f"Session: {session}")

    X_train, X_test = X
    y_train, y_test = y

    ### Few Shot Learning phase ###
    model = dummy_train(model, (X_train[0], y_train[0]))


    ### Testing phase ###

    # Define session specific dataloader with query + base_test samples
    session_query_set = torch.utils.data.TensorDataset(X_test,y_test)
    full_session_test_set = ConcatDataset([base_test_set, session_query_set])
    full_session_test_loader = DataLoader(full_session_test_set, batch_size=256, num_workers=NUM_WORKERS)

    # Create a mask function to only consider accuracy on classes presented so far
    session_classes = torch.cat((torch.arange(0,100, dtype=int), torch.unique(y_test))) 
    mask = torch.zeros((200,))
    mask[session_classes] = float('-inf')
    out_mask = lambda x: x*mask

    # Run benchmark to evaluate accuracy of this specific session
    out2pred = lambda x: torch.argmax(x, dim=-1)
    session_results = benchmark.run(dataloader = full_session_test_loader, postprocessors=[out_mask, out2pred, torch.squeeze])
    all_results.append(session_results['classification_accuracy'])

    print(f"The session accuracy is {all_results[-1]*100}%")

mean_accuracy = torch.mean(all_results)
print(f"The total mean accuracy is {mean_accuracy*100}%")