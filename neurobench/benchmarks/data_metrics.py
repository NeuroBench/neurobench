import torch

from .utils.metric_utils import check_shape

class AccumulatedMetric:
    """ Abstract class for a metric which must save state between batches.
    """

    def __init__(self):
        """ Initialize metric.
        """
        raise NotImplementedError("Subclasses of AccumulatedMetric should implement __init__")
    
    def __call__(self, model, preds, data):
        """ Process this batch of data.

        Args:
            model: A NeuroBenchModel.
            preds: A torch tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            result: the accumulated metric as of this batch.
        """
        raise NotImplementedError("Subclasses of AccumulatedMetric should implement __call__")

    def compute(self):
        """ Compute the metric score using all accumulated data.

        Returns:
            result: the final accumulated metric.
        """
        raise NotImplementedError("Subclasses of AccumulatedMetric should implement compute")


# dynamic metrics, require model, model predictions, and labels
def activation_sparsity(model, preds, data):
    """ Sparsity of model activations.
    
    Calculated as the number of zero activations over the total number
    def __init__():

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Activation sparsity.
    """
    # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
    #       Standard FF ANN should be zero (no activation sparsity)
    raise NotImplementedError("Activation sparsity not yet implemented")

    check_shape(preds, data[1])
    return model.activation_sparsity()

def multiply_accumulates(model, preds, data):
    """ Multiply-accumulates (MACs) of the model forward.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Multiply-accumulates.
    """
    # TODO: 
    #   Spiking model: number of spike activations * fanout (see snnmetrics)
    #   Recurrent layers: each connection is one MAC
    #   ANN: use PyTorch profiler
    raise NotImplementedError("Multiply-accumulates not yet implemented")

    check_shape(preds, data[1])
    macs = 0.0
    return macs

def classification_accuracy(model, preds, data):
    """ Classification accuracy of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Classification accuracy.
    """
    check_shape(preds, data[1])
    equal = torch.eq(preds, data[1])
    return torch.mean(equal.float()).item()

def MSE(model, preds, data):
    """ Mean squared error of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Mean squared error.
    """
    check_shape(preds, data[1])
    return torch.mean((preds - data[1])**2).item()

def sMAPE(model, preds, data):
    """ Symmetric mean absolute percentage error of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Symmetric mean absolute percentage error.
    """
    check_shape(preds, data[1])
    smape = 200*torch.mean(torch.abs(preds - data[1])/(torch.abs(preds)+torch.abs(data[1])))
    return torch.nan_to_num(smape, nan=200.0).item()

class r2(AccumulatedMetric):
    """ R2 Score of the model predictions.

    Currently implemented for 2D output only.
    """

    def __init__(self):
        """ Initalize metric state.

        Must hold memory of all labels seen so far.
        """
        self.x_sum_squares = 0.0
        self.y_sum_squares = 0.0
        
        self.x_labels = torch.tensor([])
        self.y_labels = torch.tensor([])

    def __call__(self, model, preds, data):
        """
        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: R2 Score.
        """
        check_shape(preds, data[1])
        self.x_sum_squares += torch.sum((data[1][:, 0] - preds[:, 0])**2).item()
        self.y_sum_squares += torch.sum((data[1][:, 1] - preds[:, 1])**2).item()
        self.x_labels = torch.cat((self.x_labels, data[1][:, 0]))
        self.y_labels = torch.cat((self.y_labels, data[1][:, 1]))

        return self.compute()

    def compute(self):
        """ Compute r2 score using accumulated data
        """
        x_denom = self.x_labels.var(correction=0)*len(self.x_labels)
        y_denom = self.y_labels.var(correction=0)*len(self.y_labels)

        x_r2 = 1 - (self.x_sum_squares/ x_denom)
        y_r2 = 1 - (self.y_sum_squares/ y_denom)

        r2 = (x_r2 + y_r2) / 2

        return r2.item()

class COCO_mAP(AccumulatedMetric):
    """ COCO mean average precision.

    Measured for event data based on Perot2020, Supplementary B (https://arxiv.org/abs/2009.13436)
        - Skips first 0.5s of each sequence
        - Bounding boxes with diagonal size smaller than 60 pixels are ignored
    """

    def __init__(self):
        """ Initalize metric state.
        """
        from metavision_ml.metrics.coco_eval import CocoEvaluator

        self.evaluator = CocoEvaluator(classes=['background'] + ["pedestrian", "two wheeler", "car"], height=360, width=640)

    def __call__(self, model, preds, data):
        """ Accumulate predictions and ground truth detections over batches.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: COCO mean average precision.
        """
        from metavision_ml.data import box_processing as box_api
        from metavision_sdk_core import EventBbox

        targets = data[1]
        video_infos = data[2]["video_infos"]
        frame_is_labeled = data[2]["frame_is_labeled"]
        skip_us = 500000

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

        for key in gt_detections:
            self.evaluator.partial_eval([np.concatenate(gt_detections[key])], [np.concatenate(dt_detections[key])])

        return self.compute()

    def compute(self):
        """ Compute COCO mAP using accumulated data.
        """
        coco_kpi = self.evaluator.accumulate()
        return coco_kpi['mean_ap']