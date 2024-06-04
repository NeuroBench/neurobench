import torch
import numpy as np
from ..utils import check_shape, make_binary_copy, single_layer_MACs
from .hooks import ActivationHook, LayerHook
from collections import defaultdict


class AccumulatedMetric:
    """Abstract class for a metric which must save state between batches."""

    def __init__(self):
        """Initialize metric."""
        raise NotImplementedError(
            "Subclasses of AccumulatedMetric should implement __init__"
        )

    def __call__(self, model, preds, data):
        """
        Process this batch of data.

        Args:
            model: A NeuroBenchModel.
            preds: A torch tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            result: the accumulated metric as of this batch.

        """
        raise NotImplementedError(
            "Subclasses of AccumulatedMetric should implement __call__"
        )

    def compute(self):
        """
        Compute the metric score using all accumulated data.

        Returns:
            result: the final accumulated metric.

        """
        raise NotImplementedError(
            "Subclasses of AccumulatedMetric should implement compute"
        )

    def reset(self):
        """
        Reset the metric state.

        This is called when the benchmark is run again, e.g. on the FSCIL task the
        benchmark     is run at the end of each session.

        """
        raise NotImplementedError(
            "Subclasses of AccumulatedMetric should implement reset"
        )


# dynamic metrics, require model, model predictions, and labels


def detect_activations_connections(model):
    """Register hooks or other operations that should be called before running a
    benchmark."""
    for hook in model.activation_hooks:
        hook.reset()
        hook.close()
    for hook in model.connection_hooks:
        hook.reset()
        hook.close()
    model.activation_hooks = []
    model.connection_hooks = []

    supported_layers = model.supported_layers

    # recurrent_supported_layers = (torch.nn.RNNBase)
    # recurr_cell_supported_layers = (torch.nn.RNNCellBase)

    act_layers = model.activation_layers()
    # Registered activation hooks
    for layer in act_layers:
        model.activation_hooks.append(ActivationHook(layer))

    con_layers = model.connection_layers()
    for flat_layer in con_layers:
        if isinstance(flat_layer, supported_layers):
            model.connection_hooks.append(LayerHook(flat_layer))


def activation_sparsity(model, preds, data):
    """
    Sparsity of model activations.

    Calculated as the number of zero activations over the total number
    of activations, over all layers, timesteps, samples in data.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Activation sparsity.

    """
    # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
    #       Standard FF ANN depends on activation function, ReLU can introduce sparsity.
    total_spike_num, total_neuro_num = 0, 0
    for hook in model.activation_hooks:
        for (
            spikes
        ) in hook.activation_outputs:  # do we need a function rather than a member
            spike_num, neuro_num = torch.count_nonzero(spikes).item(), torch.numel(
                spikes
            )
            total_spike_num += spike_num
            total_neuro_num += neuro_num

    sparsity = (
        (total_neuro_num - total_spike_num) / total_neuro_num
        if total_neuro_num != 0
        else 0.0
    )
    return sparsity


class membrane_updates(AccumulatedMetric):
    """
    Number of membrane potential updates.

    This metric can only be used for spiking models implemented with SNNTorch.

    """

    def __init__(self):
        """Init metric state."""
        self.total_samples = 0
        self.neuron_membrane_updates = defaultdict(int)

    def reset(self):
        """Reset metric state."""
        self.total_samples = 0
        self.neuron_membrane_updates = defaultdict(int)

    def __call__(self, model, preds, data):
        """
        Number of membrane updates of the model forward.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Number of membrane potential updates.

        """
        for hook in model.activation_hooks:
            for index_mem in range(len(hook.pre_fire_mem_potential) - 1):
                pre_fire_mem = hook.pre_fire_mem_potential[index_mem + 1]
                post_fire_mem = hook.post_fire_mem_potential[index_mem + 1]
                nr_updates = torch.count_nonzero(pre_fire_mem - post_fire_mem)
                self.neuron_membrane_updates[str(type(hook.layer))] += int(nr_updates)
            if len(hook.post_fire_mem_potential) > 0:
                self.neuron_membrane_updates[str(type(hook.layer))] += int(
                    torch.numel(hook.post_fire_mem_potential[0])
                )
        self.total_samples += data[0].size(0)
        return self.compute()

    def compute(self):
        """
        Compute membrane updates using accumulated data.

        Returns:
            float: Compute the total updates to each neuron's membrane potential within the model,
            aggregated across all neurons and normalized by the number of samples processed.

        """
        if self.total_samples == 0:
            return 0

        total_mem_updates = 0
        for key in self.neuron_membrane_updates:
            total_mem_updates += self.neuron_membrane_updates[key]

        total_updates_per_sample = total_mem_updates / self.total_samples
        return total_updates_per_sample


def number_neuron_updates(model, preds, data):
    """
    Number of times each neuron type is updated.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        dict: key is neuron type, value is number of updates.

    """
    # check_shape(preds, data[1])

    update_dict = {}
    for hook in model.activation_hooks:
        for spikes_batch in hook.activation_inputs:
            for spikes in spikes_batch:
                nr_updates = torch.count_nonzero(spikes)
                if str(type(hook.layer)) not in update_dict:
                    update_dict[str(type(hook.layer))] = 0
                update_dict[str(type(hook.layer))] += int(nr_updates)
    # print formatting
    print("Number of updates for:")
    for key in update_dict:
        print(key, ":", update_dict[key])
    return update_dict


def classification_accuracy(model, preds, data):
    """
    Classification accuracy of the model predictions.

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
    """
    Mean squared error of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Mean squared error.

    """
    check_shape(preds, data[1])
    return torch.mean((preds - data[1]) ** 2).item()


def sMAPE(model, preds, data):
    """
    Symmetric mean absolute percentage error of the model predictions.

    Args:
        model: A NeuroBenchModel.
        preds: A tensor of model predictions.
        data: A tuple of data and labels.
    Returns:
        float: Symmetric mean absolute percentage error.

    """
    check_shape(preds, data[1])
    smape = 200 * torch.mean(
        torch.abs(preds - data[1]) / (torch.abs(preds) + torch.abs(data[1]))
    )
    return torch.nan_to_num(smape, nan=200.0).item()


class synaptic_operations(AccumulatedMetric):
    """
    Number of synaptic operations.

    MACs for ANN ACs for SNN

    """

    def __init__(self):
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def reset(self):
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def __call__(self, model, preds, data):
        """
        Multiply-accumulates (MACs) of the model forward.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
            inputs: A tensor of model inputs.
        Returns:
            float: Multiply-accumulates.

        """
        for hook in model.connection_hooks:
            inputs = hook.inputs  # copy of the inputs, delete hooks after
            for spikes in inputs:
                # spikes is batch, features, see snntorchmodel wrappper
                # for single_in in spikes:
                if len(spikes) == 1:
                    spikes = spikes[0]
                hook.hook.remove()
                operations, spiking = single_layer_MACs(spikes, hook.layer)
                total_ops, _ = single_layer_MACs(spikes, hook.layer, total=True)
                self.total_synops += total_ops
                if spiking:
                    self.AC += operations
                else:
                    self.MAC += operations
                hook.register_hook()
        # ops_per_sample = ops / data[0].size(0)
        self.total_samples += data[0].size(0)
        return self.compute()

    def compute(self):
        if self.total_samples == 0:
            return {"Effective_MACs": 0, "Effective_ACs": 0, "Dense": 0}
        ac = self.AC / self.total_samples
        mac = self.MAC / self.total_samples
        total_synops = self.total_synops / self.total_samples
        return {"Effective_MACs": mac, "Effective_ACs": ac, "Dense": total_synops}


class r2(AccumulatedMetric):
    """
    R2 Score of the model predictions.

    Currently implemented for 2D output only.

    """

    def __init__(self):
        """
        Initalize metric state.

        Must hold memory of all labels seen so far.

        """
        self.x_sum_squares = 0.0
        self.y_sum_squares = 0.0

        self.x_labels = None
        self.y_labels = None

    def reset(self):
        """Reset metric state."""
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
        self.x_sum_squares += torch.sum((data[1][:, 0] - preds[:, 0]) ** 2).item()
        self.y_sum_squares += torch.sum((data[1][:, 1] - preds[:, 1]) ** 2).item()

        if self.x_labels is None:
            self.x_labels = data[1][:, 0]
            self.y_labels = data[1][:, 1]
        else:
            self.x_labels = torch.cat((self.x_labels, data[1][:, 0]))
            self.y_labels = torch.cat((self.y_labels, data[1][:, 1]))

        return self.compute()

    def compute(self):
        """Compute r2 score using accumulated data."""
        x_denom = self.x_labels.var(correction=0) * len(self.x_labels)
        y_denom = self.y_labels.var(correction=0) * len(self.y_labels)

        x_r2 = 1 - (self.x_sum_squares / x_denom)
        y_r2 = 1 - (self.y_sum_squares / y_denom)

        r2 = (x_r2 + y_r2) / 2

        return r2.item()


class COCO_mAP(AccumulatedMetric):
    """
    COCO mean average precision.

    Measured for event data based on Perot2020, Supplementary B (https://arxiv.org/abs/2009.13436)
        - Skips first 0.5s of each sequence
        - Bounding boxes with diagonal size smaller than 60 pixels are ignored

    """

    def __init__(self):
        """Initalize metric state."""
        from metavision_ml.metrics.coco_eval import CocoEvaluator
        from collections import defaultdict

        self.dt_detections = defaultdict(list)
        self.gt_detections = defaultdict(list)
        self.evaluator = CocoEvaluator(
            classes=["background"] + ["pedestrian", "two wheeler", "car"],
            height=360,
            width=640,
        )

    def reset(self):
        """Reset metric state."""
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
        from metavision_ml.data import box_processing as box_api
        from metavision_sdk_core import EventBbox

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
        """Compute COCO mAP using accumulated data."""
        print("Computing COCO mAP.")
        for key in self.gt_detections:
            self.evaluator.partial_eval(
                [np.concatenate(self.gt_detections[key])],
                [np.concatenate(self.dt_detections[key])],
            )
        coco_kpi = self.evaluator.accumulate()
        return coco_kpi["mean_ap"]
