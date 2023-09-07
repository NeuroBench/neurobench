from tqdm import tqdm
import numpy as np
import torch
from collections import defaultdict

from neurobench.datasets import Gen4DetectionDataLoader
from neurobench.models import NeuroBenchModel
from neurobench.benchmarks import Benchmark

from model_data.obj_detection.models import Vanilla, Vanilla_lif

from metavision_ml.detection.anchors import Anchors
from metavision_ml.detection.rpn import BoxHead
from metavision_ml.data import box_processing as box_api
from metavision_ml.metrics.coco_eval import CocoEvaluator
from metavision_sdk_core import EventBbox

# dataloader itself takes about 7 minutes for loading, with model evaluation and score calculation is about 20 minutes on i9-12900KF, RTX3080
test_set_dataloader = Gen4DetectionDataLoader(dataset_path="data/Gen 4 Multi channel",
        split="testing",
        label_map_path="neurobench/datasets/label_map_dictionary.json",
        batch_size = 4,
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

def accumulate_predictions(preds, targets, video_infos, frame_is_labeled,skip_us=500000):
    """ Accumulates predictions and ground truth detections over batches.

    Args:
        preds: model outputs, list of list of dicts, shape=(timesteps,batch,boxes)
        targets: targets, list of list of EventBbox (structured nparr), shape=(timesteps,batch) 
        video_infos (tuple): metadata of each video in batch
        frame_is_labeled (tensor): whether timestep in each sample is labeled, shape=(timestep,batch)
        skip_us (int): skips the first 500000us (0.5s) of each video, since the event feed will not have any knowledge of static objects

    Returns:
        dt_detections (dict): dictionary of video name to list of EventBbox
        gt_detections (dict): dictionary of video name to list of EventBbox

    """
    breakpoint()
    dt_detections = {}
    gt_detections = {}
    for t in range(len(targets)):
        for i in range(len(targets[t])):
            gt_boxes = targets[t][i]
            
            pred = preds[t][i]

            video_info, tbin_start, _ = video_infos[i]

            if video_info.padding or frame_is_labeled[t, i] == False:
                continue

            name = video_info.path
            if name not in dt_detections:
                dt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
            if name not in gt_detections:
                gt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
            assert video_info.start_ts == 0
            ts = tbin_start + t * video_info.delta_t

            if ts < skip_us:
                continue

            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if gt_boxes.dtype == np.float32:
                gt_boxes = box_api.box_vectors_to_bboxes(gt_boxes[:, :4], gt_boxes[:, 4], ts=ts)

            if pred['boxes'] is not None and len(pred['boxes']) > 0:
                boxes = pred['boxes'].cpu().data.numpy()
                labels = pred['labels'].cpu().data.numpy()
                scores = pred['scores'].cpu().data.numpy()
                dt_boxes = box_api.box_vectors_to_bboxes(boxes, labels, scores, ts=ts)
                dt_detections[name].append(dt_boxes)
            else:
                dt_detections[name].append(np.zeros((0), dtype=EventBbox))

            if len(gt_boxes):
                gt_boxes["t"] = ts
                gt_detections[name].append(gt_boxes)
            else:
                gt_detections[name].append(np.zeros((0), dtype=EventBbox))

    return dt_detections, gt_detections

def inference_epoch_end(outputs): 
    """ Calculation of the COCO metrics.

    Args:
        outputs (list): list of dictionaries, each dictionary contains the detections and ground truths for a video
    Returns:
        coco_kpi (dict): dictionary of COCO metrics

    """
    print('==> Start evaluation')
    dt_detections = defaultdict(list)
    gt_detections = defaultdict(list)

    for item in outputs:
        for k, v in item['gt'].items():
            gt_detections[k].extend(v)
        for k, v in item['dt'].items():
            dt_detections[k].extend(v)

    evaluator = CocoEvaluator(classes=['background'] + ["pedestrian", "two wheeler", "car"], height=360, width=640)
    for key in gt_detections:
        evaluator.partial_eval([np.concatenate(gt_detections[key])], [np.concatenate(dt_detections[key])])
    coco_kpi = evaluator.accumulate()
    return coco_kpi

# TODO: make the training scripts available in the repo, currently they are on the imec_object_detection branch

mode = "hybrid"

if mode == "ann":
    # baseline ANN RED architecture
    model = Vanilla(cin = 6, cout = 256, base = 16)
    box_coder = Anchors(num_levels=model.levels, anchor_list="PSEE_ANCHORS", variances=[0.1, 0.2])
    head = BoxHead(model.cout, box_coder.num_anchors, 3+1, 0)
    model = model.to('cuda')
    head = head.to('cuda')
    model.load_state_dict(torch.load('neurobench/examples/model_data/obj_detection/save_models/25_ann_model.pth',map_location=torch.device('cuda')))
    head.load_state_dict(torch.load('neurobench/examples/model_data/obj_detection/save_models/25_ann_pd.pth',map_location=torch.device('cuda')))

elif mode == "hybrid":
    # hybrid SNN of above architecture
    model = Vanilla_lif(cin = 6, cout = 256, base = 16)
    box_coder = Anchors(num_levels=model.levels, anchor_list="PSEE_ANCHORS", variances=[0.1, 0.2])
    head = BoxHead(model.cout, box_coder.num_anchors, 3+1, 0)
    model = model.to('cuda')
    head = head.to('cuda')
    model.load_state_dict(torch.load('neurobench/examples/model_data/obj_detection/save_models/14_hybrid_model.pth',map_location=torch.device('cuda')))
    head.load_state_dict(torch.load('neurobench/examples/model_data/obj_detection/save_models/14_hybrid_pd.pth',map_location=torch.device('cuda')))

else:
    raise ValueError("mode must be ann or hybrid")

output_val_list = []
for batch in tqdm(test_set_dataloader):
    inputs = batch[0].permute(1, 0, 2, 3, 4).to(device='cuda') # dataloader supplies batch,timestep,*; model expects timestep,batch,*
    with torch.no_grad():
        feature = model(inputs)
        loc_preds_val, cls_preds_val = head(feature)
        scores = head.get_scores(cls_preds_val)
        scores = scores.to('cpu')
        for i, feat in enumerate(feature):
            feature[i] = feature[i].to('cpu')
        inputs = inputs.to('cpu')
        loc_preds_val = loc_preds_val.to('cpu')
        preds = box_coder.decode(feature, inputs, loc_preds_val, scores, batch_size=inputs.shape[1], score_thresh=0.05,
                    nms_thresh=0.5, max_boxes_per_input=500)
        # print(preds)
        dt_dic, gt_dic = accumulate_predictions(preds, batch[1], batch[2]["video_infos"], batch[2]["frame_is_labeled"], 500000)
        output_val_list.append({'dt': dt_dic, 'gt': gt_dic})
coco_val_result = inference_epoch_end(output_val_list)
print(coco_val_result)

# batch size of inference slightly affects the results.

# Results - ANN, batch = 12
# {'mean_ap': 0.43572193867313885, 'mean_ap50': 0.7628962226722409, 'mean_ap75': 0.4300384764407489, 'mean_ap_small': -1.0, 'mean_ap_medium': 0.42157263999621397, 
# 'mean_ap_big': 0.4776233360418925, 'mean_ar': 0.5797575086266389, 'mean_ar_small': -1.0, 'mean_ar_medium': 0.5796011583452286, 'mean_ar_big': 0.5789764628578985}

# Results - Hybrid spiking, batch = 4
# {'mean_ap': 0.304865042405996, 'mean_ap50': 0.5919837857864657, 'mean_ap75': 0.27190113054105747, 'mean_ap_small': -1.0, 'mean_ap_medium': 0.3177476098097795, 
# 'mean_ap_big': 0.2957906269855653, 'mean_ar': 0.4663488181685545, 'mean_ar_small': -1.0, 'mean_ar_medium': 0.4791990086164158, 'mean_ar_big': 0.4415985913719789}

### TODO: connect to the Benchmark harness by wrapping the above model / metrics. 
### --> requires that the metrics can be aggregated and calculated at the end of all batches. also requires Benchmark cuda support.
# preprocessors = []
# postprocessors = []

# static_metrics = ["model_size"]
# data_metrics = ["COCO_mAP"]

# benchmark = Benchmark(model, test_set_dataloader, preprocessors, postprocessors, [static_metrics, data_metrics])
# results = benchmark.run()
# print(results)
