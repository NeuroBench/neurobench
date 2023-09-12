#
# NOTE: This task is still under development.
#

import wandb
from tqdm import tqdm
import torch
import os

from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import torchaudio

from torch import nn, optim, distributions as dist
import torch.nn.functional as F

import learn2learn as l2l
import copy

# from torch_mate.data.utils import IncrementalFewShot

import sys
sys.path.append("/home3/p306982/Simulations/fscil/algorithms_benchmarks/")


from neurobench.datasets import MSWC
from neurobench.datasets.IncrementalFewShot import IncrementalFewShot
from neurobench.examples.model_data.M5 import M5

from neurobench.benchmarks import Benchmark

import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for Deep Learning Parameters")
    
    parser.add_argument("--root", type=str, default="neurobench/data/MSWC", help="Root directory")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of pre-train epochs")
    parser.add_argument("--mt_iter", type=int, default=20, help="Number of meta iterations")
    parser.add_argument("--eval_iter", type=int, default=5, help="Number of evaluation iterations")
    parser.add_argument("--mt_sessions", type=int, default=8, help="Number of meta-training sessions")
    
    parser.add_argument("--mt_ways", type=int, default=10, help="Number of ways for meta-training")
    parser.add_argument("--mt_shots", type=int, default=5, help="Number of shots for meta-training")
    parser.add_argument("--mt_query_shots", type=int, default=50, help="Number of query samples for meta-training")
    parser.add_argument("--mt_pseudo_ways", type=int, default=20, help="Number of ways for pseudo-base session in meta-training")
    parser.add_argument("--mt_pseudo_shots", type=int, default=50, help="Number of shots for pseudo-base session in meta-training")
    parser.add_argument("--anil", action="store_true", help="Use ANIL version of MAML")
    parser.add_argument("--masked", action="store_true", help="Only use 100 neurons for pre-training")

    # parser.add_argument("--mt_split", nargs=2, type=int, default=(250, 250), metavar=("X", "Y"),
                        # help="The support query split to use for meta-training. Can be integer values (X, Y)")
    parser.add_argument("--no_mt_split", action="store_true", help="Don't use mt fixed support query splits")

    parser.add_argument("--eval_shots", type=int, default=5, help="Number of shots for evaluation")
    parser.add_argument("--eval_lr", type=float, default=0.001, help="Learning rate for evaluation learning")
    parser.add_argument("--eval_deep_update", action="store_true", help="Update on all layers during evaluation")
    parser.add_argument("--inner_sgd", action="store_true", help="Use SGD (instead of cross-entropy) for inner loop")
    parser.add_argument("--data_init", action="store_true", help="Use data init trick")


    parser.add_argument("--save_pre_train", action="store_true", help="Save pre trained model")
    parser.add_argument("--load_pre_train", type=str, default=None, help="Name of pre trained model to load")

    parser.add_argument("--no_wandb", action="store_true", help="Don't use wandb")

    args = parser.parse_args()
    
    return args


# NUM_WORKERS = 8
# BATCH_SIZE = 256
# PRE_TRAIN_EPOCHS = 50
# META_ITERATIONS = 10
# N_SESSIONS = 8 #excluding base session


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device == torch.device("cuda"):
    PIN_MEMORY = True
else:
    PIN_MEMORY = False

# @TODO: Clean incremental MSWC 
# @TODO: Check Pin memory for dataloaders (True when cuda)


args = parse_arguments()

ROOT = args.root
NUM_WORKERS = args.num_workers
BATCH_SIZE = args.batch_size
PRE_TRAIN_EPOCHS = args.epochs
META_ITERATIONS = args.mt_iter
N_SESSIONS = args.mt_sessions
META_WAYS = args.mt_ways
META_SHOTS = args.mt_shots
META_QUERY_SHOTS = args.mt_query_shots
META_PSEUDO_WAYS = args.mt_pseudo_ways
META_PSEUDO_SHOTS = args.mt_pseudo_shots
if args.no_mt_split:
    META_SPLITS = None
else:
    META_SPLITS = (250,250)
EVAL_ITERATIONS = args.eval_iter
EVAL_SHOTS = args.eval_shots
EVAL_LR = args.eval_lr

LOSS_FUNCTION = F.cross_entropy
if args.inner_sgd:
    FEW_SHOT_LOSS_FUNCTION = lambda x,y : F.mse_loss(x, F.one_hot(y, x.shape[-1]).float())
else:
    FEW_SHOT_LOSS_FUNCTION = F.cross_entropy
ANIL = args.anil
MASKED = args.masked
EVAL_OUT_ADAPT = not args.eval_deep_update
DATA_INIT = args.data_init

new_sample_rate = 8000
resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=new_sample_rate).to(device)
pre_proc_resample = lambda x: (resample(x[0]), x[1])
out2pred = lambda x: torch.argmax(x, dim=-1)
to_device = lambda x: (x[0].to(device), x[1].to(device))

test_loader = None

# @TODO: Improve for test_loader recreatiom
def test(test_model, mask, set=None, wandb_log="accuracy"):
    test_model.eval()
    if set is not None:
        test_loader = DataLoader(set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    else:
        base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
        test_loader = DataLoader(base_test_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    out_mask = lambda x: x - mask


    benchmark = Benchmark(test_model, metric_list=[[],["classification_accuracy"]], dataloader=test_loader, 
                          preprocessors=[to_device, pre_proc_resample], postprocessors=[out_mask, out2pred, torch.squeeze])

    pre_train_results = benchmark.run()
    test_accuracy = pre_train_results['classification_accuracy']
    if not args.no_wandb:
        wandb.log({wandb_log:test_accuracy})
    return test_accuracy


def pre_train(model):
    ### Pre-training phase ###
    base_train_set = MSWC(root=ROOT, subset="base", procedure="training")

    pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=PIN_MEMORY)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if MASKED:
        mask = torch.full((100,), 0).to(device)
    else:
        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0

    # pre_train_results = benchmark.run()
    print("PRE-TRAINING")
    for epoch in range(PRE_TRAIN_EPOCHS):
        print(f"Epoch: {epoch+1}")
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(pre_train_loader), total=len(base_train_set)//BATCH_SIZE):
            data = data.to(device)

            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = resample(data)
            output = model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = LOSS_FUNCTION(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%5==0:
            test_acc = test(model, mask)
            print(f"The test accuracy is {test_acc*100}%")
        scheduler.step()

    if args.save_pre_train:
        if MASKED:
            name = "model_ep"+str(PRE_TRAIN_EPOCHS)+"_masked"
        else:
            name = "model_ep"+str(PRE_TRAIN_EPOCHS)
        torch.save(model, os.path.join(ROOT,name))



if __name__ == '__main__':
    

    if not args.no_wandb:
        wandb.login()

        wandb_run = wandb.init(
        # Set the project where this run will be logged
        project="MSWC MetaFSCIL",
        # Track hyperparameters and run metadata
        config=args.__dict__)


    if MASKED:
        model = M5(n_input=1, n_output=100).to(device)
    else:
        model = M5(n_input=1, n_output=200).to(device)



    if args.load_pre_train:
        model = torch.load(os.path.join(ROOT, args.load_pre_train))
    else:
        pre_train(model)


        if MASKED:
            mask = torch.full((100,), 0).to(device)
        else:
            mask = torch.full((200,), float('inf')).to(device)
            mask[torch.arange(0,100, dtype=int)] = 0
        pre_train_acc = test(model, mask)
        print(f"The test accuracy at the end of pre-training is {pre_train_acc*100}%")

    ### Meta-training phase ###

    base_train_set = MSWC(root=ROOT, subset="base", procedure="training", incremental=True)

    few_shot_dataloader = IncrementalFewShot(base_train_set, n_way=META_WAYS, k_shot=META_SHOTS, query_shots=META_QUERY_SHOTS,
    first_iter_ways_shots = (META_PSEUDO_WAYS,META_PSEUDO_SHOTS),
                                incremental=True,
                                cumulative=True,
                                support_query_split=META_SPLITS,
                                samples_per_class=500)

    if ANIL:
        features = model.features
        head = model.fc1
        maml = l2l.algorithms.MAML(head, lr=EVAL_LR)
    else:
        maml = l2l.algorithms.MAML(model, lr=EVAL_LR)
        
    meta_opt = optim.Adam(maml.parameters())


    # Iteration over incremental sessions

    if MASKED:
        mask = torch.full((100,), 0).to(device)
    else:
        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0

    print("META-TRAINING")
    for iteration in range(META_ITERATIONS):
        print(f"Iteration: {iteration+1}")
        model.train()
        for session, (support, query, query_classes) in tqdm(enumerate(few_shot_dataloader), total=N_SESSIONS+1):
            # print(f"Session: {session+1}")
            

            # X_support, y_support = support
            # X_support = X_support.to(device)
            # y_support = y_support.to(device)


            # @TODO: THink if mask is needed
            

            ### Inner loop ###
            # Adaptation: Instanciate a copy of model
            learner = maml.clone()

            # support_set = TensorDataset(X_support, y_support)
            # support_loader = DataLoader(support_set, batch_size=META_WAYS, pin_memory=True)

            if ANIL:
                for X_shot, y_shot in support:
                    data = X_shot.to(device)
                    target = y_shot.to(device)
                    data = features(resample(data))
                    support_log = learner(data)
                    support_loss = FEW_SHOT_LOSS_FUNCTION(support_log.squeeze(), target) # Works with log_softmax to create crossentropy
                    learner.adapt(support_loss)
            else:
                for X_shot, y_shot in support:
                    data = X_shot.to(device)
                    target = y_shot.to(device)
                    support_log = learner(resample(data))
                    support_loss = FEW_SHOT_LOSS_FUNCTION(support_log.squeeze(), target) # Works with log_softmax to create crossentropy
                    learner.adapt(support_loss)


            ### Outer loop ###

            full_session_test_loader = DataLoader(query, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

            # Adaptation: Evaluate the effectiveness of adaptation
            meta_loss = 0
            for X_query, y_query in full_session_test_loader:
                X_query = X_query.to(device)
                y_query = y_query.to(device)
                if ANIL:
                    data = features(resample(X_query))
                    query_log = learner(data)
                else:
                    query_log = learner(resample(X_query))
                query_loss = LOSS_FUNCTION(query_log.squeeze(), y_query)
                meta_loss += query_loss/X_query.shape[0]

            meta_opt.zero_grad()
            meta_loss.backward()
            meta_opt.step()


            if session >= N_SESSIONS:
                break

        # Post meta-train base classes eval
        test_accuracy = test(model, mask)
        print(f"The test accuracy on base classes after meta-training is {test_accuracy*100}%")
        

        # Reset sampler to redefine an independant sequence of sessions
        few_shot_dataloader.reset()

    del base_train_set

    if MASKED:
        with torch.no_grad():
            new_model = M5(n_input=1, n_output=200).to(device)
            new_model.features = model.features
            new_model.fc1.weight.data[:100,:] = model.fc1.weight.data.clone()
            new_model.fc1.bias.data[:100] = model.fc1.bias.data.clone()
            model = new_model

    ### Evaluation phase ###
    print("EVALUATION")

    # Get Datasets: evaluation + all test samples from base classes to test forgetting
    eval_set = MSWC(root=ROOT, subset="evaluation")
    base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")

    all_results = []


    for eval_iter in range(EVAL_ITERATIONS):
        print(f"Evaluation Iteration: 0")
        # Base Accuracy measurement

        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0

        eval_model = copy.deepcopy(model)

        print(f"Session: 0")
        pre_train_acc = test(eval_model, mask, wandb_log="eval_accuracy")
        all_results.append(pre_train_acc)
        print(f"The base accuracy is {all_results[-1]*100}%")

        # IncrementalFewShot Dataloader used in incremental mode to generate class-incremental sessions
        few_shot_dataloader = IncrementalFewShot(eval_set, n_way=10, k_shot=EVAL_SHOTS, query_shots=100,
                                    incremental=True,
                                    cumulative=True,
                                    support_query_split=(100,100),
                                    samples_per_class=200)

        if EVAL_OUT_ADAPT:
            few_shot_optimizer = optim.SGD(eval_model.fc1.parameters(), lr=EVAL_LR)
        else:
            few_shot_optimizer = optim.SGD(eval_model.parameters(), lr=EVAL_LR)

        # Iteration over incremental sessions
        for session, (support, query, query_classes) in enumerate(few_shot_dataloader):
            print(f"Session: {session+1}")

            eval_model.train()

            # X_train, y_train = support

            ### Few Shot Learning phase ###
            # data = X_train.to(device)
            # target = y_train.to(device)

            # data = resample(data)

            # support_set = TensorDataset(X_train, y_train)
            # support_loader = DataLoader(support_set, batch_size=10, pin_memory=True)
            

            if DATA_INIT:
                with torch.no_grad():
                    # upper_bound = torch.mean(torch.max(eval_model.fc1.weight.data[:100], dim=-1)[0])
                    # lower_bound = torch.mean(torch.min(eval_model.fc1.weight.data[:100], dim=-1)[0])
                    upper_bound = 0.5
                    lower_bund = -0.5

                    new_classes = support[0][1].tolist()
                    for i, new_class in enumerate(new_classes):
                        class_data = torch.stack([shot[0][i] for shot in support])
                        class_data = resample(class_data.to(device))
                        class_representation = eval_model.features(class_data)
                        weight_vector = torch.mean(class_representation, dim=0)
                        min_class = torch.min(weight_vector)
                        max_class = torch.max(weight_vector)
                        weight_vector = weight_vector - min_class
                        gain = (upper_bound - lower_bund)/(max_class - min_class)
                        weight_vector = gain *(weight_vector) + lower_bund
                        eval_model.fc1.weight.data[new_class] = weight_vector.squeeze()



            for X_shot, y_shot in support:
                data = X_shot.to(device)
                data = resample(data)
                target = y_shot.to(device)
                output = eval_model(data)
                loss = FEW_SHOT_LOSS_FUNCTION(output.squeeze(), target)

                few_shot_optimizer.zero_grad()
                loss.backward()
                few_shot_optimizer.step()

            ### Testing phase ###

            # Define session specific dataloader with query + base_test samples
            full_session_test_set = ConcatDataset([base_test_set, query])

            # Create a mask function to only consider accuracy on classes presented so far
            session_classes = torch.cat((torch.arange(0,100, dtype=int), torch.IntTensor(query_classes))) 
            mask = torch.full((200,), float('inf')).to(device)
            mask[session_classes] = 0

            session_classes = torch.IntTensor(query_classes)
            new_mask = torch.full((200,), float('inf')).to(device)
            new_mask[session_classes] = 0

            new_class_acc = test(eval_model, new_mask, set=query, wandb_log="query_accuracy")
            print(f"The accuracy on new classes is {new_class_acc*100}%")
            # Run benchmark to evaluate accuracy of this specific session
            session_acc = test(eval_model, mask, set=full_session_test_set, wandb_log="eval_accuracy")
            all_results.append(session_acc)
            
            print(f"The session accuracy is {all_results[-1]*100}%")

        mean_accuracy = np.mean(all_results)
        if not args.no_wandb:
            wandb.log({"eval_accuracy":mean_accuracy})
        print(f"The total mean accuracy is {mean_accuracy*100}%")


        few_shot_dataloader.reset()


