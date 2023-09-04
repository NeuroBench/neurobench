#
# NOTE: This task is still under development.
#

import torch

from torch.utils.data import DataLoader, ConcatDataset
import torchaudio

from torch import nn, optim, distributions as dist
import torch.nn.functional as F

import learn2learn as l2l

# from torch_mate.data.utils import IncrementalFewShot

from neurobench.datasets import MSWC
from neurobench.datasets.IncrementalFewShot import IncrementalFewShot
from neurobench.examples.model_data.M5 import M5

from neurobench.benchmarks import Benchmark

import numpy as np

ROOT = "C:/Users/maxim/Documents/Groningen/Simulations/algorithms_benchmarks/neurobench/data/MSWC/"
NUM_WORKERS = 4
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# @TODO: CLean incremental MSWC 
# @TODO: Check Pin memory for dataloaders (True when cuda)



if __name__ == '__main__':

    dummy_train = lambda net, data: net

    model = M5(n_input=1, n_output=200).to(device)

    new_sample_rate = 8000
    resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=new_sample_rate)
    pre_proc_resample = lambda x: (resample(x[0]), x[1])

    ### Pre-training phase ###
    base_train_set = MSWC(root=ROOT, subset="base", procedure="training")

    pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
    base_test_loader = DataLoader(base_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    

    mask = torch.full((200,), float('inf'))
    mask[torch.arange(0,100, dtype=int)] = 0
    out_mask = lambda x: x - mask
    out2pred = lambda x: torch.argmax(x, dim=-1)

    benchmark = Benchmark(model, metric_list=[[],["classification_accuracy"]], dataloader=base_test_loader, preprocessors=[pre_proc_resample], postprocessors=[out_mask, out2pred, torch.squeeze])

    for batch_idx, (data, target) in enumerate(pre_train_loader):
        model.train()
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = resample(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%5==0:
            model.eval()

            pre_train_results = benchmark.run()
            # wandb pre_train_results['classification_accuracy'])


    ### Meta-training phase ###

    base_train_set = MSWC(root=ROOT, subset="base", procedure="training", incremental=True)

    few_shot_dataloader = IncrementalFewShot(base_train_set, n_way=5, k_shot=5, query_shots=50,
    first_iter_ways_shots = (20,50),
                                incremental=True,
                                cumulative=True,
                                support_query_split=(250,250),
                                samples_per_class=500)

    N_sessions = 8 #excluding base session

    maml = l2l.algorithms.MAML(model, lr=1e-3)
    opt = optim.Adam(maml.parameters())

    # Iteration over incremental sessions
    for session, (support, query, query_classes) in enumerate(few_shot_dataloader):
        print(f"Session: {session+1}")

        X_support, y_support = support


        # # Create a mask function to only consider accuracy on classes presented so far
        # session_classes = torch.IntTensor(query_classes)
        # mask = torch.full((200,), float('inf'))
        # mask[session_classes] = 0
        # out_mask = lambda x: x - mask

        

        # Adaptation: Instanciate a copy of model
        learner = maml.clone()

        ### Inner loop ###

        # Adaptation: Compute and adapt to task loss
        support_log = learner(resample(X_support))
        support_loss = F.nll_loss(support_log.squeeze(), y_support) # Works with log_softmax to create crossentropy
        learner.adapt(support_loss)


        ### Outer loop ###

        full_session_test_loader = DataLoader(query, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        # Adaptation: Evaluate the effectiveness of adaptation
        meta_loss = 0
        for X_query, y_query in full_session_test_loader:
            query_log = learner(resample(X_query))
            query_loss = F.nll_loss(query_log.squeeze(), y_query)
            meta_loss += query_loss/X_query.shape[0]

        opt.zero_grad()
        meta_loss.backward()
        opt.step()


        if session >= N_sessions:
            break


    del base_train_set

    ### Evaluation phase ###

    # Get Datasets: evaluation + all test samples from base classes to test forgetting
    eval_set = MSWC(root=ROOT, subset="evaluation")
    base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")

    # Define an arbitrary resampling as an example of pre-processor to feed to the Benchmark object
    new_sample_rate = 8000
    resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=new_sample_rate)
    pre_proc_resample = lambda x: (resample(x[0]), x[1])

    # Define benchmark object
    benchmark = Benchmark(model, metric_list=[[],["classification_accuracy"]], dataloader=None, preprocessors=[pre_proc_resample], postprocessors=[torch.nn.Identity()])
    all_results = []

    # Base Accuracy measurement
    base_test_loader = DataLoader(base_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    mask = torch.full((200,), float('inf'))
    mask[torch.arange(0,100, dtype=int)] = 0
    out_mask = lambda x: x - mask
    out2pred = lambda x: torch.argmax(x, dim=-1)

    print(f"Session: 0")
    pre_train_results = benchmark.run(dataloader = base_test_loader, postprocessors=[out_mask, out2pred, torch.squeeze])
    all_results.append(pre_train_results['classification_accuracy'])
    print(f"The base accuracy is {all_results[-1]*100}%")

    # IncrementalFewShot Dataloader used in incremental mode to generate class-incremental sessions
    few_shot_dataloader = IncrementalFewShot(eval_set, n_way=10, k_shot=5, query_shots=100,
                                incremental=True,
                                cumulative=True,
                                support_query_split=(100,100),
                                samples_per_class=200)


    # Iteration over incremental sessions
    for session, (support, query, query_classes) in enumerate(few_shot_dataloader):
        print(f"Session: {session+1}")

        X_train, y_train = support

        ### Few Shot Learning phase ###
        model = dummy_train(model, (X_train[0], y_train[0]))


        ### Testing phase ###

        # Define session specific dataloader with query + base_test samples
        full_session_test_set = ConcatDataset([base_test_set, query])
        full_session_test_loader = DataLoader(full_session_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        # Create a mask function to only consider accuracy on classes presented so far
        session_classes = torch.cat((torch.arange(0,100, dtype=int), torch.IntTensor(query_classes))) 
        mask = torch.full((200,), float('inf'))
        mask[session_classes] = 0
        out_mask = lambda x: x - mask

        # Run benchmark to evaluate accuracy of this specific session
        out2pred = lambda x: torch.argmax(x, dim=-1)
        session_results = benchmark.run(dataloader = full_session_test_loader, postprocessors=[out_mask, out2pred, torch.squeeze])
        all_results.append(session_results['classification_accuracy'])

        print(f"The session accuracy is {all_results[-1]*100}%")

    mean_accuracy = np.mean(all_results)
    print(f"The total mean accuracy is {mean_accuracy*100}%")