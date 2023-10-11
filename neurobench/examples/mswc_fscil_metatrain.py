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
from snntorch import functional as SF

import learn2learn as l2l
import copy

# from torch_mate.data.utils import IncrementalFewShot

import sys
sys.path.append("/home3/p306982/Simulations/fscil/algorithms_benchmarks/")


# from neurobench.datasets import MSWC
from neurobench.datasets.MSWC_multilingual import MSWC
from neurobench.preprocessing.speech2spikes import S2SProcessor
from neurobench.datasets.MultilingualIncFewShot import IncrementalFewShot
from neurobench.examples.model_data.M5 import M5
from neurobench.examples.model_data.sparchSNNs import SNN

from neurobench.benchmarks import Benchmark
from cl_utils import *

import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for Deep Learning Parameters")
    
    parser.add_argument("--root", type=str, default="neurobench/data/MSWC", help="Root directory")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of pre-train epochs")
    parser.add_argument("--pt_lr", type=float, default=0.01, help="Learning rate for pre-training")
    
    parser.add_argument("--spiking", action="store_true", help="Use SNN")
    parser.add_argument("--s2s_thr", type=float, default=1.0, help="Threshold for S2S Delta encoding")
    parser.add_argument("--ns_out", action="store_true", help="Use non-spiking readout")
    parser.add_argument("--hidden_size", type=int, default=256, help="Number of neurons per layer")
    parser.add_argument("--no_rec", action="store_true", help="Don't use recurrent SNNs")
    parser.add_argument("--retrain_out", action="store_true", help="Use non-spiking readout")
    parser.add_argument("--reset_out", action="store_true", help="Use non-spiking readout")
    parser.add_argument("--from_scratch", action="store_true", help="Use non-spiking readout")

    
    parser.add_argument("--n_channels", type=int, default=256, help="Number of channels")
    parser.add_argument("--dropout", action="store_true", help="Use Dropout")

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
    parser.add_argument("--eval_lr", type=float, default=0.2, help="Learning rate for evaluation learning")
    parser.add_argument("--eval_deep_update", action="store_true", help="Update on all layers during evaluation")
    parser.add_argument("--inner_mse", action="store_true", help="Use MSE (instead of cross-entropy) for inner loop")
    parser.add_argument("--data_init", action="store_true", help="Use data init trick")

    parser.add_argument("--latent_number", type=int, default=100, help="Number of last layer defining the latent features of the network")
    parser.add_argument("--freeze_below", type=str, default="output", help="Name of the last layer to freeze for inner loop")
    parser.add_argument("--reg_lambda", type=float, default=0, help="Regularization parameter for synaptic intelligence")
    parser.add_argument("--reset", type=str, default="none", choices=["zero", "random"], help="Save pre trained model")


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


args = parse_arguments()

ROOT = args.root
NUM_WORKERS = args.num_workers
BATCH_SIZE = args.batch_size

SPIKING = args.spiking

PRE_TRAIN_EPOCHS = args.epochs
PRE_TRAIN_LR = args.pt_lr
N_CHANNELS = args.n_channels
DROPOUT = args.dropout
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

if SPIKING:
    # LOSS_FUNCTION = SF.mse_count_loss(correct_rate=0.5, incorrect_rate=0.025)
    # FEW_SHOT_LOSS_FUNCTION = SF.mse_count_loss(correct_rate=0.5, incorrect_rate=0.025)
    LOSS_FUNCTION = F.cross_entropy
    # FEW_SHOT_LOSS_FUNCTION = lambda x,y : F.mse_loss(F.softmax(x, dim=1), F.one_hot(y, x.shape[-1]).float())
    FEW_SHOT_LOSS_FUNCTION = F.cross_entropy
else:
    LOSS_FUNCTION = F.cross_entropy
    if args.inner_mse:
        FEW_SHOT_LOSS_FUNCTION = lambda x,y : F.mse_loss(x, F.one_hot(y, x.shape[-1]).float())
    else:
        FEW_SHOT_LOSS_FUNCTION = F.cross_entropy
ANIL = args.anil
MASKED = args.masked
EVAL_OUT_ADAPT = not args.eval_deep_update
DATA_INIT = args.data_init

MFCC = True

new_sample_rate = 8000
if SPIKING:
    S2S = S2SProcessor(device, transpose=False)
    config_change = {"sample_rate": 48000,
                     "hop_length": 240}
    S2S.configure(threshold=args.s2s_thr, **config_change)
    pre_proc_function = S2S
    pre_proc = lambda x : S2S((x, None))[0]

elif MFCC:

    import torchaudio.transforms as T

    n_fft = 512
    win_length = None
    hop_length = 240
    n_mels = 20
    n_mfcc = 20

    mfcc_transform = T.MFCC(
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
    ).to(device)

    pre_proc = lambda  x : mfcc_transform(x).squeeze()
    pre_proc_function = lambda x: (pre_proc(x[0]), x[1])
else:
    pre_proc = torchaudio.transforms.Resample(orig_freq=48000, new_freq=new_sample_rate).to(device)
    pre_proc_function = lambda x: (pre_proc(x[0]), x[1])


if SPIKING:
    # out2pred = lambda x: torch.argmax(x.sum(0), dim=-1)
    out2pred = lambda x: torch.argmax(x, dim=-1)
else:
    out2pred = lambda x: torch.argmax(x, dim=-1)
to_device = lambda x: (x[0].to(device), x[1].to(device))

test_loader = None
reg_lambda = args.reg_lambda
LATENT_REPLAY = False
BATCH_RENORM = False
freeze_below_layer = args.freeze_below
LATENT_NUMBER = args.latent_number

def prepare_training(model, meta=False):
    rm=None

    model.saved_weights = {}
    # model.saved_norm = {'weights':{}, 'biases':{}}
    if not meta:
        pre_train_class = range(100)
        if SPIKING:
            consolidate_weights(model, model.snn[-1].W, pre_train_class)
            # consolidate_norm(model, model.snn[-1].norm, pre_train_class)
        else:
            consolidate_weights(model, model.output, pre_train_class)

    if BATCH_RENORM:
        replace_bn_with_brn(
            model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
            max_r_max=max_r_max, max_d_max=max_d_max
        )

    ### Latent Replay ###

    model.past_j = {i:0 for i in range(200)}
    model.cur_j = {i:0 for i in range(200)}
    if reg_lambda != 0:
        # the regularization is based on Synaptic Intelligence as described in the
        # paper. ewcData is a list of two elements (best parametes, importance)
        # while synData is a dictionary with all the trajectory data needed by SI
        ewcData, synData = create_syn_data(model)
        model.ewcData = ewcData
        model.synData = synData


def data_init():
    with torch.no_grad():
        # upper_bound = torch.mean(torch.max(eval_model.fc1.weight.data[:100], dim=-1)[0])
        # lower_bound = torch.mean(torch.min(eval_model.fc1.weight.data[:100], dim=-1)[0])
        upper_bound = 0.5
        lower_bund = -0.5

        new_classes = support[0][1].tolist()
        for i, new_class in enumerate(new_classes):
            class_data = torch.stack([shot[0][i] for shot in support])
            class_data = pre_proc(class_data.to(device))
            class_representation = eval_model.features(class_data)
            weight_vector = torch.mean(class_representation, dim=0)
            min_class = torch.min(weight_vector)
            max_class = torch.max(weight_vector)
            weight_vector = weight_vector - min_class
            gain = (upper_bound - lower_bund)/(max_class - min_class)
            weight_vector = gain *(weight_vector) + lower_bund
            eval_model.output.weight.data[new_class] = weight_vector.squeeze()


def inner_loop(model, support, optimizer=None, meta=None, features = torch.nn.Identity()):
    model.train()
    if not meta:
        if SPIKING:
            model.snn[:-1].eval()
            # model.snn[1].drop.train()
            freeze_below(model, "none", only_conv=False) #"snn.2.W"
            model.snn[-1].W.weight.requires_grad = True
        else:
            eval_below(model, 'output')
            freeze_below(model, "output", only_conv=False)
            

        # model.lat_features.eval()

    

    train_ep = 1

    if DATA_INIT:
        data_init()

    if reg_lambda != 0:
        init_batch(model, model.ewcData, model.synData)
        
    # freeze_up_to(model.end_features, freeze_below_layer, only_conv=False)

    if BATCH_RENORM:
        change_brn_pars(
            model, momentum=inc_update_rate, r_d_max_inc_step=0,
            r_max=max_r_max, d_max=max_d_max)



    if LATENT_REPLAY:
        # @TODO : CORRECT
        cur_class = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]))
    else:
        cur_class = support[0][1].tolist()
        model.cur_j = examples_per_class(cur_class, 200, EVAL_SHOTS)
    

    # if SPIKING:
    #     model.alpha.data[cur_class] = 0.9

    if args.reset=="zero":
        reset_weights(model, model.snn[2].W, cur_class)
    elif args.reset=="random":
        if SPIKING:
            torch.nn.init.xavier_normal_(model.snn[2].W.weight)
            #set_consolidate_weights(model, model.output)
        else:
            torch.nn.init.xavier_normal_(model.output.weight)

    set_consolidate_weights(model, model.snn[-1].W)


    cur_ep = 0

    for ep in range(train_ep):


        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0

        # computing how many patterns to inject in the latent replay layer
        if LATENT_REPLAY:
            cur_sz = train_x.size(0) // ((train_x.size(0) + rm_sz) // mb_size)
            # it_x_ep = train_x.size(0) // cur_sz
            n2inject = max(0, mb_size - cur_sz)
        else:
            n2inject = 0

        print("n2inject", n2inject)
        # @TODO : CHange data for replay case
        # it = 0
        for X_shot, y_shot in support:

            if reg_lambda !=0:
                pre_update(model, model.synData)

            # start = it * (mb_size - n2inject)
            # end = (it + 1) * (mb_size - n2inject)

            if not meta:
                optimizer.zero_grad()

            data = X_shot.to(device)

            if LATENT_REPLAY:
                lat_mb_x = rm[0][it*n2inject: (it + 1)*n2inject]
                lat_mb_y = rm[1][it*n2inject: (it + 1)*n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[start:end], lat_mb_y), 0),
                    use_cuda=use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)
            else:
                lat_mb_x = None
                target = y_shot.to(device)


            # if lat_mb_x is not None, this tensor will be concatenated in
            # the forward pass on-the-fly in the latent replay layer
            data = pre_proc(data)

            if LATENT_REPLAY:
                logits, lat_acts = model(
                data, latent_input=lat_mb_x, return_lat_acts=True)
            else:
                output = model(features(data))
                lat_acts = None
            

            # collect latent volumes only for the first ep
            # we need to store them to eventually add them into the external
            # replay memory
            if LATENT_REPLAY:
                if ep == 0:
                    lat_acts = lat_acts.cpu().detach()
                    if it == 0:
                        cur_acts = copy.deepcopy(lat_acts)
                    else:
                        cur_acts = torch.cat((cur_acts, lat_acts), 0)


            loss = FEW_SHOT_LOSS_FUNCTION(output.squeeze(), target)

            if reg_lambda !=0:
                loss += compute_ewc_loss(model, model.ewcData, lambd=reg_lambda)

            if meta:
                grads = meta.adapt(loss)
            else:
                loss.backward()
                optimizer.step()
                grads = None

            if reg_lambda !=0:
                post_update(model, model.synData, grads)

            set_consolidate_weights(model, model.snn[-1].W)



    if SPIKING:
        consolidate_weights(model, model.snn[-1].W, cur_class)
        # consolidate_norm(model, model.snn[-1].norm, cur_class)
    else:
        consolidate_weights(model, model.output, cur_class)
    if reg_lambda != 0:
        update_ewc_data(model, model.ewcData, model.synData, 0.001, 1)

    # how many patterns to save for next iter
    if LATENT_REPLAY:
        h = min(rm_sz // (i + 1), cur_acts.size(0))
        print("h", h)

        print("cur_acts sz:", cur_acts.size(0))
        idxs_cur = np.random.choice(
            cur_acts.size(0), h, replace=False
        )
        rm_add = [cur_acts[idxs_cur], train_y[idxs_cur]]
        print("rm_add size", rm_add[0].size(0))

        # replace patterns in random memory
        if i == 0:
            rm = copy.deepcopy(rm_add)
        else:
            idxs_2_replace = np.random.choice(
                rm[0].size(0), h, replace=False
            )
            for j, idx in enumerate(idxs_2_replace):
                rm[0][idx] = copy.deepcopy(rm_add[0][j])
                rm[1][idx] = copy.deepcopy(rm_add[1][j])

    if SPIKING:
        set_consolidate_weights(model, model.snn[-1].W)
        # set_consolidate_norm(model, model.snn[-1].norm)
    else:
        set_consolidate_weights(model, model.output)

    for c, n in model.cur_j.items():
        model.past_j[c] += n



# @TODO: Improve for test_loader recreatiom
def test(test_model, mask, set=None, wandb_log="accuracy", wandb_commit=True):
    test_model.eval()
    if set is not None:
        test_loader = DataLoader(set, batch_size=256, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    else:
        base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")
        test_loader = DataLoader(base_test_set, batch_size=256, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    out_mask = lambda x: x - mask


    benchmark = Benchmark(test_model, metric_list=[[],["classification_accuracy"]], dataloader=test_loader, 
                          preprocessors=[to_device, pre_proc_function], postprocessors=[out_mask, out2pred, torch.squeeze])

    pre_train_results = benchmark.run()
    test_accuracy = pre_train_results['classification_accuracy']
    if not args.no_wandb:
        wandb.log({wandb_log:test_accuracy}, commit=wandb_commit)
    return test_accuracy


def pre_train(model):
    ### Pre-training phase ###
    base_train_set = MSWC(root=ROOT, subset="base", procedure="training")

    pre_train_loader = DataLoader(base_train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=PIN_MEMORY)

    optimizer = optim.Adam(model.parameters(), lr=PRE_TRAIN_LR, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if args.retrain_out:
        freeze_below(model, "none", only_conv=False) #"snn.2.W"
        model.snn[-1].W.weight.requires_grad = True

    if args.reset_out:
        torch.nn.init.xavier_normal_(model.snn[2].W.weight)

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
            data = pre_proc(data)
            output = model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = LOSS_FUNCTION(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%1==0:
            train_acc = test(model, mask, set=base_train_set, wandb_log="train_accuracy", wandb_commit=False)
            test_acc = test(model, mask)
            print(f"The train accuracy is {train_acc*100}%")
            print(f"The test accuracy is {test_acc*100}%")
        scheduler.step()

    if args.save_pre_train:
        if SPIKING:
            name = "SpikingModel_noSoftmax_ep"+str(PRE_TRAIN_EPOCHS)+"_" + str(PRE_TRAIN_LR)+"_bs"+str(BATCH_SIZE)
            if not args.retrain_out:
                name += "_RTall"
            if args.reset_out:
                name += "_resetOut"
            if args.from_scratch:
                name += "_scratch"
            torch.save(model, os.path.join(ROOT,name))
        else:
            if DROPOUT:
                name = "model_ep"+str(PRE_TRAIN_EPOCHS)+"_" + str(PRE_TRAIN_LR)+"_chan"+str(N_CHANNELS)+"_bs"+str(BATCH_SIZE)+"_dp"
            else:
                name = "model_ep"+str(PRE_TRAIN_EPOCHS)+"_" + str(PRE_TRAIN_LR)+"_chan"+str(N_CHANNELS)+"_bs"+str(BATCH_SIZE)
            torch.save(model, os.path.join(ROOT,name))



if __name__ == '__main__':
    

    if not args.no_wandb:
        wandb.login()

        wandb_run = wandb.init(
        # Set the project where this run will be logged
        project="MSWC MetaFSCIL",
        # Track hyperparameters and run metadata
        config=args.__dict__)


    if SPIKING:
        # if args.from_scratch:
        model = SNN(
            input_shape=(args.batch_size, 201, 20),
            neuron_type="RadLIF",
            layer_sizes=[args.hidden_size, args.hidden_size, 200],
            normalization="batchnorm",
            dropout=0.1,
            bidirectional=False,
            use_readout_layer=args.ns_out,
            ).to(device)
    #     else:
        # model = torch.load(os.path.join(ROOT, "SPmodel_noSoftmax"), map_location=device)
        # model = SNN(hidden_size=args.hidden_size, rec=not args.no_rec, ns_readout=args.ns_out).to(device) #hidden_size=args.hidden_size, 
    elif MFCC:
        model = M5(n_input=20, stride=2, n_channel=N_CHANNELS, n_output=200, input_kernel=4, pool_kernel=2, latent_layer_num=LATENT_NUMBER, drop=DROPOUT).to(device)
    else:
        model = M5(n_input=1, n_output=200, latent_layer_num=LATENT_NUMBER).to(device)

    if args.load_pre_train:
        if SPIKING:
            model = torch.load(os.path.join(ROOT, args.load_pre_train), map_location=device)
            # hidden_size = load_model['forward1.weight'].size(0)
            # model = SNN(hidden_size=hidden_size, ns_readout=args.ns_out).to(device)
            # model.load_state_dict(load_model)
        else:
            load_model = torch.load(os.path.join(ROOT, args.load_pre_train), map_location=device)
            model = M5(n_input=1, n_output=200, load_model= load_model, latent_layer_num=LATENT_NUMBER).to(device)
        # model = M5(n_input=1, n_output=200, seq_model=load_model.features, output=load_model.fc1, latent_layer_num=LATENT_NUMBER).to(device)
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

    # base_train_set = MSWC(root=ROOT, subset="base", procedure="training", incremental=True)

    # few_shot_dataloader = IncrementalFewShot(base_train_set, n_way=META_WAYS, k_shot=META_SHOTS, query_shots=META_QUERY_SHOTS,
    # first_iter_ways_shots = (META_PSEUDO_WAYS,META_PSEUDO_SHOTS),
    #                             incremental=True,
    #                             cumulative=True,
    #                             support_query_split=META_SPLITS,
    #                             samples_per_class=500)

    # # if ANIL:

    # # else:
    # #     maml = l2l.algorithms.MAML(model, lr=EVAL_LR)
    # class Head(nn.Module):
    #     def __init__(self,end_features, output):
    #         super(Head, self).__init__()
    #         self.end_features = end_features
    #         self.output = output

    #     def forward(self, x):
    #         x = self.end_features(x)
    #         x = F.avg_pool1d(x, x.shape[-1])
    #         x = self.output(x.permute(0, 2, 1))
    #         return x

    # features = model.lat_features
    # head = Head(model.end_features, model.output)
    # maml = l2l.algorithms.MAML(head, lr=EVAL_LR)

    # meta_opt = optim.Adam(maml.parameters())


    # # Iteration over incremental sessions

    # if MASKED:
    #     mask = torch.full((100,), 0).to(device)
    # else:
    #     mask = torch.full((200,), float('inf')).to(device)
    #     mask[torch.arange(0,100, dtype=int)] = 0

    # print("META-TRAINING")
    # for iteration in range(META_ITERATIONS):
    #     print(f"Iteration: {iteration+1}")
    #     model.train()

    #     prepare_training(maml.module, meta=True)
    #     for session, (support, query, query_classes) in tqdm(enumerate(few_shot_dataloader), total=N_SESSIONS+1):
    #         # print(f"Session: {session+1}")

    #         ### Inner loop ###
    #         # Adaptation: Instanciate a copy of model
    #         learner = maml.clone()
            
    #         inner_loop(learner.module, support, meta=learner, features=features)



    #         ### Outer loop ###

    #         query_loader = DataLoader(query, batch_size=32, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    #         # Adaptation: Evaluate the effectiveness of adaptation
    #         meta_loss = 0
    #         for X_query, y_query in query_loader:
    #             X_query = X_query.to(device)
    #             y_query = y_query.to(device)
    #             data = features(pre_proc(X_query))
    #             query_log = learner(data)
    #             query_loss = LOSS_FUNCTION(query_log.squeeze(), y_query)
    #             meta_loss += query_loss/X_query.shape[0]

    #         meta_opt.zero_grad()
    #         meta_loss.backward()
    #         meta_opt.step()


    #         if session >= N_SESSIONS:
    #             break

    #     # Post meta-train base classes eval
    #     test_accuracy = test(model, mask)
    #     print(f"The test accuracy on base classes after meta-training is {test_accuracy*100}%")
        

    #     # Reset sampler to redefine an independant sequence of sessions
    #     few_shot_dataloader.reset()

    # del base_train_set

    # if MASKED:
    #     with torch.no_grad():
    #         new_model = M5(n_input=1, n_output=200).to(device)
    #         new_model.features = model.features
    #         new_model.output.weight.data[:100,:] = model.output.weight.data.clone()
    #         new_model.output.bias.data[:100] = model.output.bias.data.clone()
    #         model = new_model

    ### Evaluation phase ###
    print("EVALUATION")

    # Get Datasets: evaluation + all test samples from base classes to test forgetting
    # eval_set = MSWC(root=ROOT, subset="evaluation")
    base_test_set = MSWC(root=ROOT, subset="base", procedure="testing")

    all_results = []
    new_class_results = []


    for eval_iter in range(EVAL_ITERATIONS):
        print(f"Evaluation Iteration: eval_iter")
        # Base Accuracy measurement

        eval_results = []
        eval_query_results = []

        mask = torch.full((200,), float('inf')).to(device)
        mask[torch.arange(0,100, dtype=int)] = 0

        # if SPIKING:
        #     eval_model = SNN(hidden_size=hidden_size, rec=not args.no_rec, ns_readout=args.ns_out).to(device)
        #     model_dict = copy.deepcopy(model.state_dict())
        #     eval_model.load_state_dict(model_dict)
        # else:
        eval_model = copy.deepcopy(model)

        # print(f"Session: 0")
        # pre_train_acc = test(eval_model, mask, wandb_log="eval_accuracy")
        # eval_results.append(pre_train_acc)
        # print(f"The base accuracy is {eval_results[-1]*100}%")

        # IncrementalFewShot Dataloader used in incremental mode to generate class-incremental sessions
        few_shot_dataloader = IncrementalFewShot(n_way=10, k_shot=EVAL_SHOTS, 
                                    root = ROOT, 
                                    query_shots=100,
                                    incremental=True,
                                    cumulative=True,
                                    support_query_split=(100,100),
                                    samples_per_class=200)

        # if EVAL_OUT_ADAPT:
        #     few_shot_optimizer = optim.SGD(eval_model.output.parameters(), lr=EVAL_LR, momentum=0.9, weight_decay=0.0005)
        # else:
        few_shot_optimizer = optim.SGD(eval_model.parameters(), lr=EVAL_LR, momentum=0.9, weight_decay=0.0005)


        prepare_training(eval_model)

        # Iteration over incremental sessions
        for session, (support, query, query_classes) in enumerate(few_shot_dataloader):
            print(f"Session: {session+1}")

            eval_model.train()

            ### Few Shot Learning phase ###
            inner_loop(eval_model, support, few_shot_optimizer)
            

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

            new_class_acc = test(eval_model, mask, set=query, wandb_log="query_accuracy", wandb_commit=False)
            eval_query_results.append(new_class_acc)
            print(f"The accuracy on new classes is {new_class_acc*100}%")
            # Run benchmark to evaluate accuracy of this specific session
            session_acc = test(eval_model, mask, set=full_session_test_set, wandb_log="eval_accuracy")
            eval_results.append(session_acc)
            
            print(f"The session accuracy is {eval_results[-1]*100}%")

        mean_accuracy = np.mean(eval_results)
        if not args.no_wandb:
            wandb.log({"eval_accuracy":mean_accuracy})
        print(f"The total mean accuracy is {mean_accuracy*100}%")

        mean_accuracy = np.mean(eval_query_results)
        if not args.no_wandb:
            wandb.log({"query_accuracy":mean_accuracy})
        print(f"The mean query accuracy is {mean_accuracy*100}%")


        all_results.append(eval_results)
        new_class_results.append(eval_query_results)


        # few_shot_dataloader.reset()

    results = {"all": all_results, "query": new_class_results}
    import json
    if SPIKING:
        name = "eval_noreset_SPIKING_"+str(META_ITERATIONS)+"mt_"+str(EVAL_LR)+"lr.json"
    else:
        name = "eval_noreset_"+str(META_ITERATIONS)+"mt_"+str(EVAL_LR)+"lr.json"
    with open(os.path.join(ROOT,name), "w") as f:
        json.dump(results, f)

    print('DONE')



# @TODO: Change model to have latent, end features + output layer
