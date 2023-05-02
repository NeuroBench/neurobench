import os
import glob
from functools import partial
from metavision_ml.data import box_processing as box_api
from metavision_ml.data import SequentialDataLoader

# dataset_path = '/media/shenqi/study/Tue_study/Graduation/Gen4_Multi_channel'
# dataset_path = '/media/shenqi/study/Tue_study/Graduation/Gen4_Muiti_channel_small'
dataset_path = '/home/shenqi/Master_thesis/Dataset/Gen4_Multi_channel_small'
# dataset_path = '/media/shenqi/data/Gen4_multi_timesurface_FromDat'


log_dir = './log'

label_map_path = os.path.join(dataset_path, 'label_map_dictionary.json')
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

wanted_keys = ['pedestrian', 'two wheeler', 'car']
class_lookup = box_api.create_class_lookup(label_map_path, wanted_keys)
files_train = glob.glob(os.path.join(train_path, "*.h5"))
files_val = glob.glob(os.path.join(val_path, "*.h5"))
files_test = glob.glob(os.path.join(test_path, "*.h5"))

preprocess_function_name = "multi_channel_timesurface" 
delta_t = 50000
channels = 6  # histograms have two channels
num_tbins = 12
height, width = 360, 640
batch_size = 4
max_incr_per_pixel = 5
array_dim = [num_tbins, channels, height, width]
array_dim_val = [3, channels, height, width]

load_boxes_fn = partial(box_api.load_boxes, num_tbins=num_tbins, class_lookup=class_lookup, min_box_diag_network=60)

class seq_dataloader:
    def __init__(self,) :
        
        self.in_channels = channels
        self.wanted_keys = wanted_keys
        self.height = height
        self.width = width
        self.seq_dataloader_train = SequentialDataLoader(files_train, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=4, padding=True, preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})
        self.seq_dataloader_val = SequentialDataLoader(files_val, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=4, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})
        self.seq_dataloader_test = SequentialDataLoader(files_test, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=4, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel})

