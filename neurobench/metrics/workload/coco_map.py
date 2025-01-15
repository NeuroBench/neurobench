import torch
from neurobench.metrics.abstract.workload_metric import AccumulatedMetric
from collections import defaultdict
import numpy as np

try:
    from metavision_ml.metrics.coco_eval import CocoEvaluator
    from metavision_ml.data import box_processing as box_api
    from metavision_sdk_core import EventBbox

    METAVISION_AVAILABLE = True
except ImportError:
    METAVISION_AVAILABLE = False


class CocoMap(AccumulatedMetric):
    """
    COCO mean average precision.

    Measured for event data based on Perot2020, Supplementary B (https://arxiv.org/abs/2009.13436)
        - Skips first 0.5s of each sequence
        - Bounding boxes with diagonal size smaller than 60 pixels are ignored

    """

    def __init__(self):
        """
        Initialize the CocoMap metric.

        Raises:
            ImportError: If the `metavision_ml` and `metavision_sdk_core` packages are not installed on a supported platform.

        """

        if not METAVISION_AVAILABLE:
            raise ImportError(
                "metavision_ml and metavision_sdk_core are required for COCO_mAP metric. "
                "Please install them on a supported platform."
            )

        super().__init__(requires_hooks=False)
        self.dt_detections = defaultdict(list)
        self.gt_detections = defaultdict(list)
        self.evaluator = CocoEvaluator(
            classes=["background"] + ["pedestrian", "two wheeler", "car"],
            height=360,
            width=640,
        )

    def reset(self):
        """
        Reset metric state.

        Clears all accumulated detections and reinitializes the `CocoEvaluator`.

        """
        self.dt_detections = defaultdict(list)
        self.gt_detections = defaultdict(list)
        self.evaluator = CocoEvaluator(
            classes=["background"] + ["pedestrian", "two wheeler", "car"],
            height=360,
            width=640,
        )

    def __call__(self, model, preds, data):
        """
        Accumulate predictions and ground truth detections over batches.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: COCO mean average precision.

        """

        targets = data[1]
        video_infos = data[2]["video_infos"]
        frame_is_labeled = data[2]["frame_is_labeled"]
        skip_us = 500000

        for t in range(len(targets)):
            for i in range(len(targets[t])):
                gt_boxes = targets[t][i]

                pred = preds[t][i]

                video_info, tbin_start, _ = video_infos[i]

                if video_info.padding or frame_is_labeled[t, i] is False:
                    continue

                name = video_info.path
                if name not in self.dt_detections:
                    self.dt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
                if name not in self.gt_detections:
                    self.gt_detections[name] = [np.zeros((0), dtype=box_api.EventBbox)]
                assert video_info.start_ts == 0
                ts = tbin_start + t * video_info.delta_t

                if ts < skip_us:
                    continue

                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = gt_boxes.cpu().numpy()
                if gt_boxes.dtype == np.float32:
                    gt_boxes = box_api.box_vectors_to_bboxes(
                        gt_boxes[:, :4], gt_boxes[:, 4], ts=ts
                    )

                if pred["boxes"] is not None and len(pred["boxes"]) > 0:
                    boxes = pred["boxes"].cpu().data.numpy()
                    labels = pred["labels"].cpu().data.numpy()
                    scores = pred["scores"].cpu().data.numpy()
                    dt_boxes = box_api.box_vectors_to_bboxes(
                        boxes, labels, scores, ts=ts
                    )
                    self.dt_detections[name].append(dt_boxes)
                else:
                    self.dt_detections[name].append(np.zeros((0), dtype=EventBbox))

                if len(gt_boxes):
                    gt_boxes["t"] = ts
                    self.gt_detections[name].append(gt_boxes)
                else:
                    self.gt_detections[name].append(np.zeros((0), dtype=EventBbox))

        return 0.0  # too heavy to compute every iteration

    def compute(self):
        """
        Compute COCO mAP using accumulated data.

        Returns:
            float: COCO mean average precision.

        """
        print("Computing COCO mAP.")
        for key in self.gt_detections:
            self.evaluator.partial_eval(
                [np.concatenate(self.gt_detections[key])],
                [np.concatenate(self.dt_detections[key])],
            )
        coco_kpi = self.evaluator.accumulate()
        return coco_kpi["mean_ap"]
