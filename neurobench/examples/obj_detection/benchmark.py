import torch

import spikingjelly

from neurobench.datasets import Gen4DetectionDataLoader
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

from obj_det_model import Vanilla, Vanilla_lif

from metavision_ml.detection.anchors import Anchors
from metavision_ml.detection.rpn import BoxHead

import argparse

parser = argparse.ArgumentParser(description='NeuroBench benchmark for object detection models')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')
parser.add_argument('--mode', type=str, default="ann", help='mode of the model, ann or hybrid')
args = parser.parse_args()

# dataloader itself takes about 7 minutes for loading, with model evaluation and score calculation is about 20 minutes on i9-12900KF, RTX3080
test_set_dataloader = Gen4DetectionDataLoader(dataset_path="data/Gen 4 Multi channel",
        split="testing",
        label_map_path="neurobench/datasets/label_map_dictionary.json",
        batch_size = args.batch_size,
        num_tbins = 12,
        preprocess_function_name="multi_channel_timesurface",
        delta_t=50000,
        channels=6,  # multichannel six channels
        height=360,
        width=640,
        max_incr_per_pixel=5,
        class_selection=["pedestrian", "two wheeler", "car"],
        num_workers=2)

# Evaluation pipeline written and models trained by Shenqi Wang (wang69@imec.be) and Guangzhi Tang (guangzhi.tang@imec.nl) at imec.

class ObjDetectionModel(NeuroBenchModel):
    def __init__(self, net, box_coder, head):
        super(ObjDetectionModel, self).__init__(net)
        self.net = net
        self.box_coder = box_coder
        self.head = head

    def __call__(self, batch):
        self.net.eval()
        inputs = batch.permute(1, 0, 2, 3, 4).to(device='cuda') # dataloader supplies batch,timestep,*; model expects timestep,batch,*
        with torch.no_grad():
            feature = self.net(inputs)
            loc_preds_val, cls_preds_val = self.head(feature)
            scores = self.head.get_scores(cls_preds_val)
            scores = scores.to('cpu')
            for i, feat in enumerate(feature):
                feature[i] = feature[i].to('cpu')
            inputs = inputs.to('cpu')
            loc_preds_val = loc_preds_val.to('cpu')
            preds = box_coder.decode(feature, inputs, loc_preds_val, scores, batch_size=inputs.shape[1], score_thresh=0.05,
                        nms_thresh=0.5, max_boxes_per_input=500)
        return preds

    def __net__(self):
        # returns only network, not box_coder and head
        return self.net

# Loading the model
mode = args.mode # "ann" or "hybrid
if mode == "ann":
    # baseline ANN RED architecture
    model = Vanilla(cin = 6, cout = 256, base = 16)
    box_coder = Anchors(num_levels=model.levels, anchor_list="PSEE_ANCHORS", variances=[0.1, 0.2])
    head = BoxHead(model.cout, box_coder.num_anchors, 3+1, 0)
    model = model.to('cuda')
    head = head.to('cuda')
    model.load_state_dict(torch.load('neurobench/examples/obj_detection/model_data/save_models/25_ann_model.pth',map_location=torch.device('cuda')))
    head.load_state_dict(torch.load('neurobench/examples/obj_detection/model_data/save_models/25_ann_pd.pth',map_location=torch.device('cuda')))
elif mode == "hybrid":
    # hybrid SNN of above architecture
    model = Vanilla_lif(cin = 6, cout = 256, base = 16)
    box_coder = Anchors(num_levels=model.levels, anchor_list="PSEE_ANCHORS", variances=[0.1, 0.2])
    head = BoxHead(model.cout, box_coder.num_anchors, 3+1, 0)
    model = model.to('cuda')
    head = head.to('cuda')
    model.load_state_dict(torch.load('neurobench/examples/obj_detection/model_data/save_models/14_hybrid_model.pth',map_location=torch.device('cuda')))
    head.load_state_dict(torch.load('neurobench/examples/obj_detection/model_data/save_models/14_hybrid_pd.pth',map_location=torch.device('cuda')))
else:
    raise ValueError("mode must be ann or hybrid")

model = ObjDetectionModel(model, box_coder, head)

# add activation modules for hybrid models
if mode == "hybrid":
    model.add_activation_module(spikingjelly.activation_based.neuron.BaseNode)

# Evaluation
preprocessors = []
postprocessors = []

static_metrics = ["model_size", "connection_sparsity"]
workload_metrics = ["activation_sparsity", "COCO_mAP", "synaptic_operations"]


benchmark = Benchmark(model, test_set_dataloader, preprocessors, postprocessors, [static_metrics, workload_metrics])
results = benchmark.run()
print(results)

# batch size of inference slightly affects the results.

# Results - ANN, batch = 4
# {'model_size': 91314912, 'connection_sparsity': 0.0, 'activation_sparsity': 0.6339577418819095, 'COCO_mAP': 0.4286601323956029, 'synaptic_operations': {'Effective_MACs': 248423062860.16266, 'Effective_ACs': 0.0, 'Dense': 284070730752.0}}
# Results - Hybrid, batch = 4
# {'model_size': 12133872, 'connection_sparsity': 0.0, 'activation_sparsity': 0.6130047485397788, 'COCO_mAP': 0.27111120859281557, 'synaptic_operations': {'Effective_MACs': 37520084211.538666, 'Effective_ACs': 559864693.7093333, 'Dense': 98513107968.0}}