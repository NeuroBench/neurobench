import os
import torch
import pickle
import numpy as np
from scipy.io import loadmat
import torch.nn as nn
from sklearn.metrics import r2_score
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import data_processing, early_stop
from model import ANN_model
from data import dataset, dataloader


data_path = "../dataset"
assert os.path.isdir(data_path), 'Update data_path to the folder that contains the dataset'
postpr_data_path = f"{data_path}/postpr_data/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
max_epoch = 50
time_shift = 1

batch_size = 256
hidden_neuron1 = 32
hidden_neuron2 = 48

summation = 1
regenerate = False


if __name__ == "__main__":
    """
    The main function of training ANN model
    """
    total_data = dataset.find_all_data_file(data_path)  # Find all available files in our dataset
    for file_no in range(len(total_data)):
        file_path = os.path.join(data_path, total_data[file_no])
        print("File_name : {}".format(total_data[file_no]))
        fname = total_data[file_no].split('.')[0]

        try:
            if regenerate:
                raise Exception("regenerate postprocessed data...")
        
            with open(os.path.join(f'{postpr_data_path}', 'input', f'{fname}.pkl'), 'rb') as f:
                Yenc = pickle.load(f)
                print("Successfully loaded train samples from:", f'{postpr_data_path}', 'input', f'{fname}.pkl')

            with open(os.path.join(f'{postpr_data_path}', 'label', f'{fname}.pkl'), 'rb') as f:
                Xenc = pickle.load(f)
                print("Successfully loaded train samples from:", f'{postpr_data_path}', 'label', f'{fname}.pkl')

        except:
            Yenc, Xenc = dataset.data_processing(file_path, 
                                                 device=device, 
                                                 summation=summation, 
                                                 advance=0.016, 
                                                 bin_width=0.016*3, 
                                                 postpr_data_path=postpr_data_path, 
                                                 filename=fname)

        for time_shift in range(2):
            print("Summation = {}, Time_shift = {}".format(summation, time_shift))

            Yenc1, Xenc1 = data_processing.dataset_time_shift(time_shift, Yenc, Xenc)
            dataset_train, dataset_val, dataset_test = data_processing.data_split(Yenc1, Xenc1, train_ratio = 0.8, val_ratio = 0.1)
            final_results = torch.zeros((3, 6), device=device)

            for run in range(3):
                best_score = torch.zeros((3, 6), device=device)
                net = ANN_model.ANNModel(input_dim=Yenc1.shape[0])
                net.to(device)
                net.train()

                trainloader, valoader, testloader = dataloader.data_loader(dataset_train, dataset_val, dataset_test, batch_size)

                optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
                criterion = nn.MSELoss()

                Learning_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

                val_loss_hist = []
                test_loss_hist = []
                loss_hist = []
                min_loss = 100
                early_stopping = early_stop.EarlyStopping(patience=7, verbose=False)


                for epoch in tqdm(range(max_epoch), desc="Training Epoch"):
                    train_batch = iter(trainloader)
                    train_loss_hist = []
                    r2score_train = 0
                    train_count = 0

                    for i, (data, target) in enumerate(train_batch):
                        #print(data.shape, target.shape)
                        data = data.to(device)
                        target = target.to(device)

                        pre = net(data.view(batch_size, -1))
                        #print(pre.shape, target.shape)
                        loss_val = criterion(pre, target)

                        optimizer.zero_grad()
                        loss_val.backward()
                        optimizer.step()

                        train_loss_hist.append(loss_val.item())
                        r2score_train += r2_score(target.detach().cpu().numpy(), pre.detach().cpu().numpy())
                        train_count += 1
                        # print('Run: {}, Epoch: {}, Step: {}, Loss: {}'.format(run, epoch, i, loss_val.item()))

                        if (i + 1) % 10 == 0:
                            net.eval()
                            r2score_val = 0
                            total_val = 0
                            val_loss_val = 0
                            with torch.no_grad():
                                for val_data, val_target in iter(valoader):
                                    val_data = val_data.to(device)
                                    val_target = val_target.to(device)
                                    val_pre = net(val_data.view(batch_size, -1))

                                    val_loss_val += criterion(val_pre, val_target)
                                    r2score_val += r2_score(val_target.detach().cpu().numpy(), val_pre.detach().cpu().numpy())
                                    total_val += 1

                            val_loss_hist.append(val_loss_val.item() / total_val)
                            if val_loss_hist[-1] < min_loss:
                                min_loss = val_loss_hist[-1]
                            # print(' Validation loss: {}, R2_score: {}'.format(val_loss_hist[-1], r2score_val / total_val))

                            net.train()

                    loss_hist.append(sum(train_loss_hist) / train_count)
                    # print('Training loss: {}, R2_score: {}'.format(loss_hist[-1], r2score_train / train_count))

                    net.eval()
                    r2score_test = 0
                    total_test = 0
                    test_loss_val = 0
                    with torch.no_grad():
                        for test_data, test_target in iter(testloader):
                            test_data = test_data.to(device)
                            test_target = test_target.to(device)
                            test_pre = net(test_data.view(batch_size, -1))

                            test_loss_val += criterion(test_pre, test_target)
                            r2score_test += r2_score(test_target.detach().cpu().numpy(), test_pre.detach().cpu().numpy())
                            total_test += 1

                        test_loss_hist.append(test_loss_val.item() / total_test)
                        # print(' Test loss: {}, R2_score: {}'.format(test_loss_hist[-1], r2score_test / total_test))

                    early_stopping(val_loss_hist[-1], net)

                    if early_stopping.early_stop or epoch == max_epoch-1:
                        final_results[run, 0], final_results[run, 1] = loss_hist[-1], r2score_train / train_count
                        final_results[run, 2], final_results[run, 3] = test_loss_hist[-1], r2score_test / total_test
                        final_results[run, 4], final_results[run, 5] = val_loss_hist[-1], r2score_val / total_val
                        break


                    Learning_scheduler.step()
                    net.train()

            #print(final_results)
            final_mean = torch.mean(final_results, dim=0)
            print("Training_Loss = {}, Training_R2 = {}, Test_Loss = {}, Test_R2 = {}, Validation_Loss = {}, Validation_R2 = {}".format(
                final_mean[0], final_mean[1], final_mean[2], final_mean[3], final_mean[4], final_mean[5]))

            with open("./data.txt", "a") as f:
                f.write("File_name = " + str(total_data[file_no]) + "\n")
                f.write("Summation = " + str(summation) + " " + "Time_shift = " + str(time_shift) + "\n")

                lst_data = [str(final_mean[1]), str(final_mean[3]), str(final_mean[5]), str(final_mean[0]),
                       str(final_mean[2]), str(final_mean[4])]
                f.write(" ".join(lst_data) + "\n")

                f.close()
