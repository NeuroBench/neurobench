import os
import random
import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader



class ClassificationDataset(Dataset):
    def __init__(self, args, mode):
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size
        self.quantization_size = [args.sample_size//args.T,1,1]
        self.w, self.h = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]
        
        save_file_name = f"{args.dataset}_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"
        save_file = os.path.join(args.path, save_file_name)
        
        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}.")
            
    def __getitem__(self, index):
        sparse_tensor, target = self.samples[index]
        sample = sample.sparse_resize_(
            (self.T, sample.size(1), sample.size(2), self.C), 3, 1
        ).to_dense().permute(0,3,1,2)
        sample = T.Resize((64,64), T.InterpolationMode.NEAREST)(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, data_dir, save_file):
        raise NotImplementedError("The method build_dataset has not been implemented.")


class NCARSClassificationDataset(ClassificationDataset):
    def __init__(self, args, mode="train"):
        super().__init__(args, mode)

    def build_dataset(self, data_dir, save_file):
        classes_dir = [os.path.join(data_dir, class_name) for class_name in os.listdir(data_dir)]
        samples = []
        for class_id,class_dir in enumerate(classes_dir):
            self.files = [os.path.join(class_dir, time_seq_name) for time_seq_name in os.listdir(class_dir)]
            target = class_id
            
            print(f'Building the class number {class_id+1}')
            pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)
            for file_name in self.files:
                print(f"Processing {file_name}...")
                video = PSEELoader(file_name)
                events = video.load_delta_t(self.sample_size)

                if events.size == 0:
                    print("Empty sample.")
                    continue
                
                events['t'] -= events['t'][0]
                coords = torch.from_numpy(
                    structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.float32))

               # Bin the events on T timesteps
                coords = torch.floor(coords/torch.tensor(self.quantization_size)) 
                coords[:, 1].clamp_(min=0, max=self.quantized_h-1)
                coords[:, 2].clamp_(min=0, max=self.quantized_w-1)

                # TBIN computations
                tbin_size = self.quantization_size[0] / self.tbin

                # get for each ts the corresponding tbin index
                tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
                # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
                tbin_feats = ((events['p']+1) * (tbin_coords+1)) - 1

                feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*self.tbin)

                sparse_tensor = torch.sparse_coo_tensor(
                    coords.t().to(torch.int32), 
                    feats,
                    (self.T, self.quantized_h, self.quantized_w, self.C),
                )

                sparse_tensor = sparse_tensor.coalesce().to(torch.bool)

                samples.append((sparse_tensor, target))
                pbar.update(1)
                
            pbar.close()

        return samples

class GEN1ClassificationDataset(ClassificationDataset):
    def __init__(self, args, mode="train"):
        super().__init__(args, mode)
        self.undersample_cars_percent = args.undersample_cars_percent

    def build_dataset(self, data_dir, save_file):
        # Remove duplicates (.npy and .dat)
        self.files = [os.path.join(data_dir, time_seq_name[:-9]) for time_seq_name in os.listdir(data_dir)
                        if time_seq_name[-3:] == 'npy']

        print('Building the dataset')
        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)
        samples = []
        for file_name in self.files:
            print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)
            
            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            
            curr_ts = 0

            for box in boxes:
                if box['ts'] != curr_ts:
                    # Video loading
                    video.seek_time(box['ts']-self.sample_size)
                    events = video.load_delta_t(self.sample_size)

                    if events.size == 0:
                        continue
                    curr_ts = box['ts']
                target = box['class_id']

                # Undersample cars
                if self.mode == "train" and target == 0:
                    if random.random() > self.undersample_cars_percent:
                        continue

                events['t'] -= events['t'][0]
                all_coords = torch.from_numpy(
                    structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

                # Keep only objects
                obj_idxs = (
                    (all_coords[:,1]>box['y']) &
                    (all_coords[:,1]<(box['y']+box['h'])) & 
                    (all_coords[:,2]>box['x']) & 
                    (all_coords[:,2]<(box['x']+box['w']))
                )
                coords = all_coords[obj_idxs,:]

                # Bin the events on T timesteps
                coords = torch.floor(coords/torch.tensor(self.quantization_size)) 
                coords[:, 1].clamp_(min=0, max=self.quantized_h-1)
                coords[:, 2].clamp_(min=0, max=self.quantized_w-1)

                # TBIN computations
                tbin_size = self.quantization_size[0] / self.tbin

                # get for each ts the corresponding tbin index
                tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
                # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
                tbin_feats = ((events['p']+1) * (tbin_coords+1)) - 1

                feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*self.tbin)

                sparse_tensor = torch.sparse_coo_tensor(
                    coords.t().to(torch.int32), 
                    feats,
                    (self.T, self.quantized_h, self.quantized_w, self.C),
                )

                sparse_tensor = sparse_tensor.coalesce().to(torch.bool)

                samples.append((sparse_tensor, target))

                # Oversample pedestrians with an horizontal mirror
                if self.mode == "train" and target == 1:
                    coords[:,1] = self.w-coords[:,1]-1
                        sparse_tensor = torch.sparse_coo_tensor(
                        coords.t().to(torch.int32), 
                        feats,
                        (self.T, self.quantized_h, self.quantized_w, self.C),
                    )

                    sparse_tensor = sparse_tensor.coalesce().to(torch.bool)

                    samples.append((sparse_tensor, target))
                        
            pbar.update(1)
        pbar.close()

        return samples
