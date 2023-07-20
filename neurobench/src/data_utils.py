import os
import glob
from functools import partial
from metavision_ml.data import box_processing as box_api
from metavision_ml.data import SequentialDataLoader


from utils import DataParams, TrainingParams

dataset_path = '/app/prophesee_gen4/prophese_gen4_mct_hdf5/Gen4 Multi channel timesurface/'
log_dir = './log'

label_map_path = os.path.join(dataset_path, 'label_map_dictionary.json')
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

wanted_keys = DataParams.classes_to_use#['pedestrian', 'two wheeler', 'car']
class_lookup = box_api.create_class_lookup(label_map_path, wanted_keys)
# subsetting to run the model, to be removed
files_train = glob.glob(os.path.join(train_path, "*.h5"))[:100]

# filtering only 30 files
files_val = glob.glob(os.path.join(val_path, "*.h5"))[:32]
files_test = glob.glob(os.path.join(test_path, "*.h5"))


preprocess_function_name = "multi_channel_timesurface" 
delta_t = DataParams.t_delta
channels = 6  # histograms have two channels
num_tbins = DataParams.num_tbins
height, width = DataParams.height, DataParams.width
batch_size = DataParams.batch_size
max_incr_per_pixel = 5
array_dim = [num_tbins, channels, height, width]
num_workers = 4*TrainingParams.n_gpus

load_boxes_fn = partial(box_api.load_boxes, class_lookup=class_lookup, min_box_diag_network=60)



class seq_dataloader:
    def __init__(self,) :
        
        self.in_channels = channels
        self.wanted_keys = wanted_keys
        self.height = height
        self.width = width
        self.seq_dataloader_train = SequentialDataLoader(files_train, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=num_workers, padding=True, preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel}).cuda()
        self.seq_dataloader_val = SequentialDataLoader(files_val, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=num_workers, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel}).cuda()
        self.seq_dataloader_test = SequentialDataLoader(files_test, delta_t, preprocess_function_name, array_dim, load_labels=load_boxes_fn,
                                      batch_size=batch_size, num_workers=num_workers, padding=True,preprocess_kwargs={"max_incr_per_pixel": max_incr_per_pixel}).cuda()