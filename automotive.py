from neurobench.datasets import Gen4DetectionDataLoader

from neurobench.benchmarks import Benchmark

from tqdm import tqdm

'''
Note: The following files have no corresponding labels in the precomputed datasets, and
should be removed manually from the test folder.

moorea_2019-06-17_test_01_000_1769500000_1829500000.h5 
moorea_2019-06-14_000_549500000_609500000.h5
moorea_2019-06-14_000_1220500000_1280500000.h5
moorea_2019-06-26_test_02_000_1708500000_1768500000.h5
'''


test_set_dataloader = Gen4DetectionDataLoader(dataset_path="data/Gen 4 Histograms",
        split="testing",
        label_map_path="neurobench/datasets/label_map_dictionary.json",
        batch_size = 12,
        num_tbins = 12,
        preprocess_function_name="histo",
        delta_t=50000,
        channels=2,  # histograms have two channels
        height=360,
        width=640,
        max_incr_per_pixel=5,
        class_selection=["pedestrian", "two wheeler", "car"],
        num_workers=4)

# All samples can be loaded in ~2 minutes
for batch in tqdm(test_set_dataloader):
        pass