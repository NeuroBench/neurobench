# other imports
import torch
import numpy as np
import pickle as pkl
import os
import logging
from hashlib import md5
from PIL import Image
from torch.utils.data import Subset
import copy
from utils import * 

exp_name = "ar1-free_v1"
comment = "with latent replay"
use_cuda = True
init_lr = 0.0001
inc_lr = 0.0005 # 0.00005
mb_size = 32
init_train_ep = 4
inc_train_ep = 30
# init_update_rate = 0.01
# inc_update_rate = 0.00005
max_r_max = 1.25
max_d_max = 0.5
# inc_step = 4.1e-05
rm_sz = 1000
momentum = 0.9
l2 = 0.0005
freeze_below_layer = "lat_features.9.weight"
reg_lambda = 0.3 #1000 ? 

#%%

seq_model = torch.load("trained_models/seq_model")

latent_layer_num = 11
from models.CNN import CNN

model = CNN(latent_layer_num=latent_layer_num).to(device=device)

# Optimizer setup
optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
)
criterion = torch.nn.CrossEntropyLoss()

Nmax_classes = 11

#%%

cur_class = [0,1,2,3,4]
# def init(model, cur_class = [0,1,2,3,4]):
tot_it_step = 0
rm=None

model.saved_weights = {}
consolidate_weights(model, cur_class)

model.past_j = {i:0 for i in range(Nmax_classes)}
model.cur_j = {i:0 for i in range(Nmax_classes)}
if reg_lambda != 0:
    # the regularization is based on Synaptic Intelligence as described in the
    # paper. ewcData is a list of two elements (best parametes, importance)
    # while synData is a dictionary with all the trajectory data needed by SI
    ewcData, synData = create_syn_data(model)

# Optimizer setup
optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
)
criterion = torch.nn.CrossEntropyLoss()

def get_PT_data(PT_trainset, Nsamples):

    pre_train_x = np.zeros((Nsamples,1, 2,64,64))
    pre_train_y = np.zeros((Nsamples))
    for i in range(Nsamples):
        index = i*5000%64161
        x, y = PT_trainset[index]
        pre_train_x[i] = x
        pre_train_y[i] = y

def get_incremental_data(label, datasets_dict, Nsamples):
    trainset, testset = datasets_dict[label]
    inc_train_y = np.full((Nsamples,), label)
    inc_train_x = np.zeros((Nsamples,1, 2,64,64))
    for i, (x, y) in enumerate(trainset):
        inc_train_x[i] = x
        if i==Nsamples-1:
            break

    return inc_train_x, inc_train_y

def train(model, train_x, train_y, cur_class, i):

    ## MULTI LAYER TRICKS ##
    if reg_lambda != 0:
        init_batch(model, ewcData, synData)

    # we freeze the layer below the replay layer since the first batch
    freeze_up_to(model, freeze_below_layer, only_conv=False)

    ## END ##

    if i == 1:
        # change_brn_pars(
        #     model, momentum=inc_update_rate, r_d_max_inc_step=0,
        #     r_max=max_r_max, d_max=max_d_max)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
        )


    if i == 0:
        # cur_class = [int(o) for o in set(train_y)]
        model.cur_j = examples_per_class(train_y, 11)
    else:
        # cur_class = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]), 11)

    # print("----------- batch {0} -------------".format(i))
    # print("train_x shape: {}, train_y shape: {}"
    #         .format(train_x.shape, train_y.shape))

    model.train()
    model.lat_features.eval()


    reset_weights(model, cur_class) # GOES WITH CONSOLIDATE WEIGHTS

    ## PADDING AND SHUFFLING NOT USEFUL ###
    if i == 0:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)
    shuffle_in_unison([train_x, train_y], in_place=True)


    cur_ep = 0

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if i == 0:
        train_ep = init_train_ep
    else:
        train_ep = inc_train_ep

    for ep in range(train_ep):


        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0

        # Latent R: computing how many patterns to inject in the latent replay layer
        if i > 0:
            cur_sz = train_x.size(0) // ((train_x.size(0) + rm_sz) // mb_size)
            it_x_ep = train_x.size(0) // cur_sz
            n2inject = max(0, mb_size - cur_sz)
        else:
            n2inject = 0
        print("total sz:", train_x.size(0) + rm_sz)
        print("n2inject", n2inject)
        print("it x ep: ", it_x_ep)

        for it in range(it_x_ep):

            # @TODO: CHECK Pre_update to also not put output layer in there
            if reg_lambda !=0:
                pre_update(model, synData)

            start = it * (mb_size - n2inject)
            end = (it + 1) * (mb_size - n2inject)

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

            if i == 0:
                lat_mb_x = None
                y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            else:
                lat_mb_x = rm[0][it*n2inject: (it + 1)*n2inject]
                lat_mb_y = rm[1][it*n2inject: (it + 1)*n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[start:end], lat_mb_y), 0),
                    use_cuda=use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

            # if lat_mb_x is not None, this tensor will be concatenated in
            # the forward pass on-the-fly in the latent replay layer
            logits, lat_acts = model(
                x_mb, latent_input=lat_mb_x, return_lat_acts=True)

            # collect latent volumes only for the first ep
            # we need to store them to eventually add them into the external
            # replay memory
            if ep == 0:
                lat_acts = lat_acts.cpu().detach()
                if it == 0:
                    cur_acts = copy.deepcopy(lat_acts)
                else:
                    cur_acts = torch.cat((cur_acts, lat_acts), 0)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            if reg_lambda !=0:
                loss += compute_ewc_loss(model, ewcData, lambd=reg_lambda)
            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            if reg_lambda !=0:
                post_update(model, synData)

            acc = correct_cnt.item() / \
                    ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 1 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

            # Log scalar values (scalar summary) to TB
            tot_it_step +=1
            # writer.add_scalar('train_loss', ave_loss, tot_it_step)
            # writer.add_scalar('train_accuracy', acc, tot_it_step)

        cur_ep += 1