import torch
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset
import torchaudio.transforms as T

from neurobench.datasets import MSWC
from neurobench.datasets.MSWC_IncrementalLoader import IncrementalFewShot
from neurobench.examples.mswc_fscil.M5 import M5
from neurobench.models import TorchModel

from neurobench.benchmarks import Benchmark
from neurobench.preprocessing import MFCCProcessor

from cl_utils import *

ROOT = "data/MSWC/"
NUM_WORKERS = 8
BATCH_SIZE = 256
NUM_REPEATS = 1
PRE_TRAIN = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"):
    PIN_MEMORY = True
else:
    PIN_MEMORY = False

# Define MFCC pre-processing 
n_fft = 512
win_length = None
hop_length = 240
n_mels = 20
n_mfcc = 20

mfcc = MFCCProcessor(
    sample_rate=48000,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
        "f_min": 20,
        "f_max": 4000,
    },
    device = device
)

squeeze = lambda x: (x[0].squeeze(), x[1])
out2pred = lambda x: torch.argmax(x, dim=-1)
to_device = lambda x: (x[0].to(device), x[1].to(device))


def pre_train(model):

    base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
    pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=PIN_MEMORY)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(50):
        print(f"Epoch: {epoch+1}")
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(pre_train_loader), total=len(base_train_set)//BATCH_SIZE):
            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data, target = mfcc((data,target))
            output = model(data.squeeze())

            loss = F.cross_entropy(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
    
    del base_train_set
    del pre_train_loader

if __name__ == '__main__':


    if PRE_TRAIN:
        ### Pre-training phase ###
        model = M5(n_input=20, stride=2, n_channel=256, 
                n_output=200, input_kernel=4, pool_kernel=2, drop=True).to(device)

        pre_train(model)
        model = TorchModel(model)

    else:
        ### Loading Pre-trained model ###

        model = M5(n_input=20, stride=2, n_channel=256, 
                n_output=200, input_kernel=4, pool_kernel=2, drop=True).to(device)
        load_dict = torch.load("neurobench/examples/mswc_fscil/model_data/mswc_mfcc_cnn", 
                               map_location=device).state_dict()
        model.load_state_dict(load_dict)
        model = TorchModel(model)


    for eval_iter in range(NUM_REPEATS):
        print(f"Evaluation Iteration: 0")
        ### Evaluation phase ###

        eval_model = copy.deepcopy(model)

        eval_accs = []
        query_accs = []
        act_sparsity = []
        syn_ops_dense = []
        syn_ops_macs = []

        # Get base test set for evaluation
        base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
        test_loader = DataLoader(base_test_set, batch_size=256, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # Define an arbitrary resampling as an example of pre-processor to feed to the Benchmark object
        eval_model.net.eval()

        # Metrics
        static_metrics = ["model_size", "connection_sparsity"]
        data_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

        # Define benchmark object
        benchmark_all_test = Benchmark(eval_model, metric_list=[static_metrics, data_metrics], dataloader=test_loader, 
                            preprocessors=[to_device, mfcc, squeeze], postprocessors=[])

        benchmark_new_classes = Benchmark(eval_model, metric_list=[[],["classification_accuracy"]], dataloader=test_loader,
                            preprocessors=[to_device, mfcc, squeeze], postprocessors=[])

        # Define specific post-processing with masking on the base classes
        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0
        out_mask = lambda x: x - mask

        # Run session 0 benchmark on base classes
        print(f"Session: 0")
        pre_train_results = benchmark_all_test.run(postprocessors=[out_mask, out2pred, torch.squeeze])
        print("Base results:", pre_train_results)
        eval_accs.append(pre_train_results['classification_accuracy'])
        act_sparsity.append(pre_train_results['activation_sparsity'])
        syn_ops_dense.append(pre_train_results['synaptic_operations']['Dense'])
        syn_ops_macs.append(pre_train_results['synaptic_operations']['Effective_MACs'])
        print(f"The base accuracy is {eval_accs[-1]*100}%")

        # IncrementalFewShot Dataloader used in incremental mode to generate class-incremental sessions
        few_shot_dataloader = IncrementalFewShot(n_way=10, k_shot=5, 
                                    root = ROOT,
                                    query_shots=100,
                                    support_query_split=(100,100),
                                    samples_per_class=200)

        # Saves shifted versions of 
        eval_model.net.saved_weights = {}
        pre_train_class = range(100)
        consolidate_weights(eval_model.net, pre_train_class)

        few_shot_optimizer = torch.optim.SGD(eval_model.net.parameters(), lr=0.3, momentum=0.9, weight_decay=0.0005)

        # Iteration over incremental sessions
        for session, (support, query, query_classes) in enumerate(few_shot_dataloader):
            print(f"Session: {session+1}")

            ### Few Shot Learning phase ###
            eval_model.net.train()
            #eval_model.lat_features.eval()
            freeze_below(eval_model.net, "output", only_conv=False)
            eval_below(eval_model.net, "output")

            cur_class = support[0][1].tolist()
            eval_model.net.cur_j = examples_per_class(cur_class, 200, 5)

            # Update weigts over successive shots
            for X_shot, y_shot in support:
                few_shot_optimizer.zero_grad()

                data, target = mfcc((X_shot.to(device), y_shot.to(device)))
                data = data.squeeze()

                output = eval_model(data)

                loss = F.cross_entropy(output.squeeze(), target)
                loss.backward()
                few_shot_optimizer.step()

            # Mean-shift weights and save them 
            consolidate_weights(eval_model.net, cur_class)
            set_consolidate_weights(eval_model.net)


            ### Testing phase ###
            eval_model.net.eval()

            # Define session dataloaders for query and query + base_test samples
            query_loader = DataLoader(query, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            
            full_session_test_set = ConcatDataset([base_test_set, query])
            full_session_test_loader = DataLoader(full_session_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

            # Create a mask function to only consider accuracy on classes presented so far
            session_classes = torch.cat((torch.arange(0,100, dtype=int), torch.IntTensor(query_classes))) 
            mask = torch.full((200,), float('inf')).to(device)
            mask[session_classes] = 0
            out_mask = lambda x: x - mask

            # Run benchmark to evaluate accuracy of this specific session
            session_results = benchmark_all_test.run(dataloader = full_session_test_loader, postprocessors=[out_mask, out2pred, torch.squeeze])
            print("Session results:", session_results)
            # session_acc = session_results['classification_accuracy']
            # print(f"The session accuracy is {session_acc*100}%")
            eval_accs.append(session_results['classification_accuracy'])
            act_sparsity.append(session_results['activation_sparsity'])
            syn_ops_dense.append(session_results['synaptic_operations']['Dense'])
            syn_ops_macs.append(session_results['synaptic_operations']['Effective_MACs'])
            print(f"Session accuracy: {session_results['classification_accuracy']*100} %")

            # Run benchmark on query classes only
            query_results = benchmark_new_classes.run(dataloader = query_loader, postprocessors=[out_mask, out2pred, torch.squeeze])
            print(f"Accuracy on new classes: {query_results['classification_accuracy']*100} %")
            query_accs.append(query_results['classification_accuracy'])
            # query_acc = query_results['classification_accuracy']
            # print(f"The accuracy on new classes is {query_acc*100}%")

        # mean_accuracy = np.mean(eval_accs)
        # print(f"The total mean accuracy is {mean_accuracy*100}%")

        # Print all data
        print(f"Eval Accs: {eval_accs}")
        print(f"Query Accs: {query_accs}")
        print(f"Act Sparsity: {act_sparsity}")
        print(f"Syn Ops Dense: {syn_ops_dense}")
        print(f"Syn Ops MACs: {syn_ops_macs}")
