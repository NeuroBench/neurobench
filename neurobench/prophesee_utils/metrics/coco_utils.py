import torch
import torchvision.ops.boxes as box_ops

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# modified from https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/metrics/coco_eval.py

def coco_eval(gts, detections, height, width, labelmap=("car", "pedestrian")):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]

    dataset, results = _to_coco_format(gts, detections, categories, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def _to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []
    ann_id = 0

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections), 1):
        images.append({
            "date_captured": "2019",
            "file_name": "n.a",
            "id": image_id,
            "license": 1,
            "url": "",
            "height": height,
            "width": width
            })
        
        target_nb_boxes = gt['labels'].shape[0]
        
        for i in range(target_nb_boxes):
            bbox = gt['boxes'][i,:]
            class_id = gt['labels'][i]
            converted_bbox = box_ops.box_convert(bbox, "xyxy", "xywh")
            x, y, w, h = converted_bbox
            area = w * h
            
            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "category_id": int(class_id + 1),
                "id": ann_id
            }
            annotations.append(annotation)
            ann_id += 1
            
        pred_nb_boxes = pred['labels'].shape[0]
        
        for i in range(pred_nb_boxes):
            bbox = pred['boxes'][i,:]
            class_id = pred['labels'][i]
            score = pred['scores'][i]
            converted_bbox = box_ops.box_convert(bbox, "xyxy", "xywh")
            x, y, w, h = converted_bbox
            
            image_result = {
                'image_id': image_id,
                'category_id': int(class_id + 1),
                'score': float(score),
                'bbox': [x, y, w, h],
            }
            results.append(image_result)

    dataset = {
        "info": {},
        "licenses": [],
        "type": 'instances',
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return dataset, results