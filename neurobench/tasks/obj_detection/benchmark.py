from neurobench.datasets import Gen4DetectionDataLoader
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

test_set_dataloader = Gen4DetectionDataLoader(dataset_path="data/Gen 4 Multi channel",
        split="testing",
        label_map_path="neurobench/datasets/label_map_dictionary.json",
        batch_size = 12,
        num_tbins = 12,
        preprocess_function_name="multi_channel_timesurface",
        delta_t=50000,
        channels=6,  # multichannel six channels
        height=360,
        width=640,
        max_incr_per_pixel=5,
        class_selection=["pedestrian", "two wheeler", "car"],
        num_workers=4)

# dataloader output is (data, labels, kwargs)

# whole test set loads in about 7 minutes
# test_set_dataloader.cuda()

preprocessors = []
postprocessors = []

### TODO: define model, load model, writeup metrics ###


# load model
net = ...
box_coder = ...
head = ...
model = ObjDetectionModel(net, box_coder, head)

static_metrics = ["COCO_mAP"]
data_metrics = ["classification_accuracy"]

benchmark = Benchmark(model, test_set_loader, preprocessors, postprocessors, [static_metrics, data_metrics])
results = benchmark.run()
print(results)
