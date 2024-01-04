import os
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset

from neurobench.datasets import MSWC
from neurobench.datasets.MSWC_IncrementalLoader import IncrementalFewShot
from neurobench.examples.mswc_fscil.M5 import M5
from neurobench.models import TorchModel

from neurobench.examples.mswc_fscil.sparchSNNs import SNN
from neurobench.examples.mswc_fscil.sparchSNNs import RadLIFLayer

from neurobench.benchmarks import Benchmark
from neurobench.preprocessing import MFCCProcessor, S2SProcessor

import argparse


squeeze = lambda x: (x[0].squeeze(), x[1])
out2pred = lambda x: torch.argmax(x, dim=-1)
to_device = lambda x: (x[0].to(device), x[1].to(device))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser")
    
    parser.add_argument("--spiking", action='store_true',  help="Run pre-training")

    args = parser.parse_args()

    return args


args = parse_arguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_DIR = "neurobench/examples/mswc_fscil/model_data/"
ROOT = "data/MSWC/"
NUM_WORKERS = 8 if device == torch.device("cuda") else 0
BATCH_SIZE = 256
NUM_REPEATS = 1
SPIKING = args.spiking
PRE_TRAIN = False
EVAL_EPOCHS = 1
EVAL_SHOTS = 5
EVAL_WAYS = 10
EPOCHS = 50 # if pre training from scratch

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

if SPIKING:
    encode = S2SProcessor(device, transpose=True)
    config_change = {"sample_rate": 48000,
                     "hop_length": 240}
    encode.configure(threshold=1.0, **config_change)
else:
    encode = MFCCProcessor(
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


def test(test_model, mask, set=None):
    test_model.eval()

    if set is not None:
        test_loader = DataLoader(set, batch_size=256, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    else:
        base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
        test_loader = DataLoader(base_test_set, batch_size=256, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    out_mask = lambda x: x - mask

    benchmark = Benchmark(test_model, metric_list=[[],["classification_accuracy"]], dataloader=test_loader, 
                          preprocessors=[to_device, encode, squeeze], postprocessors=[out_mask, out2pred, torch.squeeze])

    pre_train_results = benchmark.run()
    test_accuracy = pre_train_results['classification_accuracy']

    return test_accuracy


def pre_train(model):
    base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
    pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=PIN_MEMORY)

    mask = torch.full((200,), float('inf')).to(device)
    mask[torch.arange(0,100, dtype=int)] = 0

    if SPIKING:
        lr = 0.001
    else:
        lr = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}")

        model.train()

        for _, (data, target) in tqdm(enumerate(pre_train_loader), total=len(base_train_set)//BATCH_SIZE):
            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data, target = encode((data,target))
            output = model(data.squeeze())

            loss = F.cross_entropy(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            train_acc = test(model, mask, set=base_train_set)
            test_acc = test(model, mask)

            print(f"The train accuracy is {train_acc*100}%")
            print(f"The test accuracy is {test_acc*100}%")

        scheduler.step()

    del base_train_set
    del pre_train_loader


if __name__ == '__main__':

    if PRE_TRAIN:
        ### Pre-training phase ###
        if SPIKING:
            model = SNN(
                input_shape=(BATCH_SIZE, 201, 20),
                neuron_type="RadLIF",
                layer_sizes=[1024, 1024, 200],
                normalization="batchnorm",
                dropout=0.1,
                bidirectional=False,
                use_readout_bias=False,
                use_readout_layer=True,
                ).to(device)
            
            pre_train(model)

            model = TorchModel(model)
            model.add_activation_module(RadLIFLayer)
        else:
            model = M5(n_input=20, stride=2, n_channel=256, 
                    n_output=200, input_kernel=4, pool_kernel=2, drop=True).to(device)

            pre_train(model)

            model = TorchModel(model)
    else:
        ### Loading Pre-trained model ###

        if SPIKING:
            model = SNN(
                input_shape=(BATCH_SIZE, 201, 20),
                neuron_type="RadLIF",
                layer_sizes=[1024, 1024, 200],
                normalization="batchnorm",
                dropout=0.1,
                bidirectional=False,
                use_readout_layer=True,
                ).to(device)
            
            model = torch.load(os.path.join(MODEL_SAVE_DIR, "mswc_rsnn_proto"),
                                map_location=device)
            model = TorchModel(model)
            
            model.add_activation_module(RadLIFLayer)
        else:
            model = M5(n_input=20, stride=2, n_channel=256, 
                    n_output=200, input_kernel=4, pool_kernel=2, drop=True).to(device)
            model = torch.load(os.path.join(MODEL_SAVE_DIR,"mswc_cnn_proto"),
                                map_location=device)
            model = TorchModel(model)

    all_evals = []
    all_query = []
    all_act_sparsity = []
    all_syn_ops_dense = []
    all_syn_ops_macs = []
    all_syn_ops_acs = []

    base_train_set = MSWC(root=ROOT, subset="base", procedure="training")
    train_loader = DataLoader(base_train_set, batch_size=500, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if SPIKING:
        output = model.net.snn[-1].W
        proto_out = nn.Linear(output.weight.shape[1], 200, bias=True).to(device)
        proto_out.weight.data = output.weight.data
    else:
        output = model.net.output
        proto_out = nn.Linear(512, 200, bias=True).to(device)
        proto_out.weight.data = output.weight.data

    # Compute prototype weights for base classes
    print("Computing prototype weights for base classes")
    for data, target in tqdm(train_loader):
        data, target = encode((data.to(device), target.to(device)))
        data = data.squeeze()
        class_id = target[0]

        if SPIKING:
            features = data
            for layer in model.net.snn[:-1]:
                features = layer(features)

            mean = torch.sum(features, dim=[0,1])/500
            proto_out.weight.data[class_id] = 2*mean
            proto_out.bias.data[class_id] = -torch.matmul(mean, mean.t())/features.shape[1]

        else:
            features = model.net(data, features_out=True)

            mean = torch.sum(features, dim=0)/500
            proto_out.weight.data[class_id] = 2*mean
            proto_out.bias.data[class_id] = -torch.matmul(mean, mean.t())

        del data
        del features
        del mean

    if SPIKING:
        model.net.snn[-1].W = proto_out
    else:
        model.net.output = proto_out

    del base_train_set
    del train_loader

    # Get base test set for evaluation
    base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")

    for eval_iter in range(NUM_REPEATS):
        print(f"Evaluation Iteration: 0")
        ### Evaluation phase ###

        eval_model = copy.deepcopy(model)

        eval_accs = []
        query_accs = []
        act_sparsity = []
        syn_ops_dense = []
        syn_ops_macs = []
        syn_ops_acs = []

        test_loader = DataLoader(base_test_set, batch_size=256, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # Define an arbitrary resampling as an example of pre-processor to feed to the Benchmark object
        eval_model.net.eval()

        # Metrics
        static_metrics = ["model_size", "connection_sparsity"]
        workload_metrics = ["classification_accuracy", "activation_sparsity", "synaptic_operations"]

        # Define benchmark object
        benchmark_all_test = Benchmark(eval_model, metric_list=[static_metrics, workload_metrics], dataloader=test_loader, 
                            preprocessors=[to_device, encode, squeeze], postprocessors=[])

        benchmark_new_classes = Benchmark(eval_model, metric_list=[[],["classification_accuracy"]], dataloader=test_loader,
                            preprocessors=[to_device, encode, squeeze], postprocessors=[])

        # Define specific post-processing with masking on the base classes
        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0
        out_mask = lambda x: x - mask

        # Run session 0 benchmark on base classes
        print(f"Session: 0")

        pre_train_results = benchmark_all_test.run(postprocessors=[out_mask, F.softmax, out2pred, torch.squeeze])
        
        print("Base results:", pre_train_results)
        
        eval_accs.append(pre_train_results['classification_accuracy'])
        act_sparsity.append(pre_train_results['activation_sparsity'])
        syn_ops_dense.append(pre_train_results['synaptic_operations']['Dense'])
        syn_ops_macs.append(pre_train_results['synaptic_operations']['Effective_MACs'])
        syn_ops_acs.append(pre_train_results['synaptic_operations']['Effective_ACs'])
        
        print(f"The base accuracy is {eval_accs[-1]*100}%")

        # IncrementalFewShot Dataloader used in incremental mode to generate class-incremental sessions
        few_shot_dataloader = IncrementalFewShot(n_way=EVAL_WAYS, k_shot=EVAL_SHOTS, 
                                    root = ROOT,
                                    query_shots=100,
                                    support_query_split=(100,100),
                                    samples_per_class=200)

        # Iteration over incremental sessions
        for session, (support, query, query_classes) in enumerate(few_shot_dataloader):
            print(f"Session: {session+1}")

            # Define benchmark object
            benchmark_all_test = Benchmark(eval_model, metric_list=[static_metrics, workload_metrics], dataloader=test_loader, 
                                preprocessors=[to_device, encode, squeeze], postprocessors=[])

            benchmark_new_classes = Benchmark(eval_model, metric_list=[[],["classification_accuracy"]], dataloader=test_loader,
                                preprocessors=[to_device, encode, squeeze], postprocessors=[])

            ### Computing new Prototypical Weights ###
            data = None
            
            for X_shot, y_shot in support:
                if data is None:
                    data = X_shot
                    target = y_shot
                else:
                    data = torch.cat((data,X_shot), 0)
                    target = torch.cat((target,y_shot), 0)

            data, target = encode((data.to(device), target.to(device)))
            data = data.squeeze()

            if SPIKING:
                features = eval_model.net.snn[0](data)
                features = eval_model.net.snn[1](features)

                for index, class_id in enumerate(query_classes[-EVAL_WAYS:]):
                    mean = torch.sum(features[[i*EVAL_WAYS+index for i in range(EVAL_SHOTS)]], dim=[0,1])/EVAL_SHOTS
                    eval_model.net.snn[-1].W.weight.data[class_id] = 2*mean
                    eval_model.net.snn[-1].W.bias.data[class_id] = -torch.matmul(mean, mean.t())/(features.shape[1])
            else:
                features = eval_model.net(data, features_out=True)

                for index, class_id in enumerate(query_classes[-EVAL_WAYS:]):
                    mean = torch.sum(features[[i*EVAL_WAYS+index for i in range(EVAL_SHOTS)]], dim=0)/EVAL_SHOTS
                    eval_model.net.output.weight.data[class_id] = 2*mean
                    eval_model.net.output.bias.data[class_id] = -torch.matmul(mean, mean.t())

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
            session_results = benchmark_all_test.run(dataloader = full_session_test_loader, postprocessors=[out_mask, F.softmax, out2pred, torch.squeeze])
            print("Session results:", session_results)

            eval_accs.append(session_results['classification_accuracy'])
            act_sparsity.append(session_results['activation_sparsity'])
            syn_ops_dense.append(session_results['synaptic_operations']['Dense'])
            syn_ops_macs.append(session_results['synaptic_operations']['Effective_MACs'])
            syn_ops_acs.append(pre_train_results['synaptic_operations']['Effective_ACs'])
            print(f"Session accuracy: {session_results['classification_accuracy']*100} %")

            # Run benchmark on query classes only
            query_results = benchmark_new_classes.run(dataloader = query_loader, postprocessors=[out_mask, F.softmax, out2pred, torch.squeeze])
            print(f"Accuracy on new classes: {query_results['classification_accuracy']*100} %")
            query_accs.append(query_results['classification_accuracy'])

        all_evals.append(eval_accs)
        all_query.append(query_accs)
        all_act_sparsity.append(act_sparsity)
        all_syn_ops_dense.append(syn_ops_dense)
        all_syn_ops_macs.append(syn_ops_macs)
        all_syn_ops_acs.append(syn_ops_acs)

        mean_accuracy = np.mean(eval_accs)
        print(f"The total mean accuracy is {mean_accuracy*100}%")

        # Print all data
        print(f"Eval Accs: {eval_accs}")
        print(f"Query Accs: {query_accs}")
        print(f"Act Sparsity: {act_sparsity}")
        print(f"Syn Ops Dense: {syn_ops_dense}")
        print(f"Syn Ops MACs: {syn_ops_macs}")
