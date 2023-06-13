"""
"""
import numpy as np
import os

from torch.utils.data import DataLoader, Subset
import torch

def utility_function():
    ...

def find_all_data_file(data_path):
    """
    Find all of the data files of dataset

    :param data_path: File path of dataset
    :return: Names of each data file
    """
    total_data = []
    datalist = os.listdir(data_path)
    for item in datalist:
        if item.endswith('.mat'):
            total_data.append(item)
    return total_data

def save_results(data, train_ratio, delay, fname, path):

    with open(os.path.join(path, "data.txt"), "a") as f:
        f.write("File_name = " + str(fname) + "\n")
        f.write("Train_ratio = " + str(train_ratio) + " " + "Time_shift = " +
                str(delay) + "\n")

        lst_data = [str(data[1]), str(data[3]), str(data[5]), str(data[0]),
                    str(data[2]), str(data[4])]
        f.write(" ".join(lst_data) + "\n")

        f.close()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_label = False

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.best_label = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_label = True
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #         torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def operation_calculation(net, model_type: str, sparsity: list=[], spikes=[]):
    # Sparsity List contains the following:
    # [layer3's sparsity, input sparsity, layer1's sparsity, layer2's sparsity]

    multiplies, adds = 0, 0
    if model_type == "ANN":
        for param in net.named_parameters():
            print(param[0], param[1].size()[::-1])
            local_dim = param[1].size()[::-1]
            if "weight" in param[0]:
                temp = 1
                for dim in local_dim:
                    print("DIM: {}".format(dim))
                    temp *= dim
                
                if "batchnorm" not in param[0]:
                    if "1" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[1]))
                        temp *= (spikes[1].mean())
                    elif "2" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[2]))
                        temp *= (spikes[2].mean())
                    elif "3" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[3]))
                        temp *= (spikes[3].mean())
                
                multiplies += temp
                if "batchnorm" not in param[0]:
                    adds += temp
            elif "bias"in param[0]:
                temp = 1
                for dim in local_dim:
                    temp *= dim

                adds += temp
            print("Current Mult: {}.\nCurrent Add: {}".format(multiplies, adds))
    elif model_type == "SNN":
        ns = net.num_step
        for param in net.named_parameters():
            print(param[0], param[1].size()[::-1])
            local_dim = param[1].size()[::-1]
            if "weight" in param[0]:
                temp = 1
                for dim in local_dim:
                    print("DIM: {}".format(dim))
                    temp *= dim
                
                if "batchnorm" not in param[0]:
                    if "1" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[1])) # Sparsiy here implies (No of spikes/timestep) zero elements on average
                        temp *= (1-sparsity[1])*ns
                    elif "2" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[2]))
                        temp *= (1-sparsity[2])*ns
                    elif "3" in param[0]:
                        print("Local Sparsity is: {}".format(sparsity[3]))
                        temp *= (1-sparsity[3])*ns
                    adds += temp    # Addition Operations for the Weight's being applied to the Spiking Neuron
                else:
                    multiplies += temp*ns   # This is for Batch Norm's division
                    
            elif "bias" in param[0]:
                temp = 1
                for dim in local_dim:
                    temp *= dim
                temp *= ns
                adds += temp
                if "batchnorm" not in param[0]:
                    multiplies += temp
                    adds += temp

            print("Current Mult: {}.\nCurrent Add: {}".format(multiplies, adds))
    else:
        print("Invalid Input Model Type. Can only be ANN or SNN, got {} instead!".format(model_type))

    print("Total Number of Multiply Ops is: {}".format(multiplies))
    print("Total Number of Add Ops is: {}".format(adds))

    return multiplies, adds

def sparsity_calculation(model_type: str, model_outputs: tuple, num_step=1):
    layers_sparsity = [0 for _ in range(len(model_outputs))]
    
    if model_type == "ANN":
        for i in range(len(model_outputs)):
            sparsity = 1 - (len(torch.nonzero(model_outputs[i])) / torch.numel(model_outputs[i]))
            if i == 0:
                print("Final output sparsity is: {}".format(sparsity))
            elif i == 1:
                print("Input Sparsity is: {}".format(sparsity))
            else:
                print("Layer-{}'s Sparsity is: {}".format(i-1, sparsity))
            layers_sparsity[i] = sparsity
    elif model_type == "SNN":
        for i in range(len(model_outputs)):
            if i == 1:
                sparsity = 1 - (len(torch.nonzero(model_outputs[i])) / torch.numel(model_outputs[i]))
                print("Input Sparsity is: {}".format(sparsity))
            elif i == len(model_outputs)-1:
                continue
            else:
                local_sparsity = []
                for step in range(num_step):
                    sparsity = 1 - (len(torch.nonzero(model_outputs[i][step])) / torch.numel(model_outputs[i][step]))
                    local_sparsity.append(sparsity)
                sparsity = sum(local_sparsity)/len(local_sparsity)
                if i == 0:
                    print("Final output sparsity is: {}".format(sparsity))
                else:
                    print("Layer-{}'s Sparsity is: {}".format(i-1, sparsity))
            layers_sparsity[i] = sparsity
    else:
        print("Invalid Model Type. Should be either ANN or SNN, got {}".format(model_type))
    return layers_sparsity

def input_operations_calculation(data):
    print("Input Operation Calculation:")
    print(data.size())
    print(torch.mean(data))
