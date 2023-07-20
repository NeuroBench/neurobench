from typing import Any, List, Dict
from itertools import chain
import gc

from torch import Tensor
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import torchvision.models.detection._utils as det_utils
import torchvision.ops.boxes as box_ops

import spikingjelly

from models.backbone import BackBone_OD
from models.backbone_vgg import DetectionBackbone
from models.anchorgen import GridSizeDefaultBoxGenerator, filter_boxes
from models.ssd import SSDHead

from eval_utils.coco_utils import coco_eval


class ObjDetectionModule(pl.LightningModule):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.width, self.height = self.args.image_shape
        self.label_map = args.target_classes
        self.lr = args.lr

        self.backbone = DetectionBackbone(args).to('cuda')
        self.anchor_generator = GridSizeDefaultBoxGenerator(args.aspect_ratios, args.min_ratio, args.max_ratio).to('cuda')

        assert len(self.backbone.out_channels) == len(self.anchor_generator.aspect_ratios)

        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.ssd_head = SSDHead(self.backbone.out_channels, num_anchors, args.num_classes).to('cuda')
        self.box_coder = det_utils.BoxCoder(weights=args.box_coder_weights)
        self.proposal_matcher = det_utils.SSDMatcher(args.iou_threshold)
        

    def prepare_features_targets(self, features, targets, frame_is_labeled):
        targets  = [[self.create_targets(y) for y in x] for x in targets]
        frame_is_labeled = frame_is_labeled.bool()
        mask = frame_is_labeled.view(-1)
        features = [item[mask] for item in features]
        targets = [
            [targets[r][c] for c in range(len(frame_is_labeled[0])) if frame_is_labeled[r][c]]
            for r in range(len(frame_is_labeled))]
        
        targets = list(chain.from_iterable(targets))
        del frame_is_labeled, mask
        gc.collect()
        return features, targets

    def create_targets(self, boxes):
        try:
            torch_boxes = torch.from_numpy(structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32)).to('cuda')
            # keep only last instance of every object per target
            _,unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True) # keep last unique objects
            unique_indices = np.flip(-(unique_indices+1))
            torch_boxes = torch_boxes[[*unique_indices]]
            
            torch_boxes[:, 2:] += torch_boxes[:, :2] # implicit conversion to xyxy
            torch_boxes[:, 0::2].clamp_(min=0, max=360)
            torch_boxes[:, 1::2].clamp_(min=0, max=640)
            
            # valid idx = width and height of GT bbox aren't 0
            valid_idx = (torch_boxes[:,2]-torch_boxes[:,0] != 0) & (torch_boxes[:,3]-torch_boxes[:,1] != 0)
            torch_boxes = torch_boxes[valid_idx, :]
            
            torch_labels = torch.from_numpy(boxes['class_id'].view(np.int32)).to(torch.long).to('cuda')
            torch_labels = torch_labels[[*unique_indices]]
            torch_labels = torch_labels[valid_idx]
            torch_labels = torch_labels - 1
            del valid_idx, unique_indices 
        except:
            print("Exception occurred in processing targets")
            torch_boxes = torch.Tensor([[]])
            torch_labels = torch.Tensor([])

        
        return {'boxes': torch_boxes, 'labels': torch_labels}

    def forward(self, events):
        features = self.backbone(events)
        head_outputs = self.ssd_head(features)
        return features, head_outputs
    
    def compute_loss(self, targets: List[Dict[str, Tensor]], 
                     head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        num_foreground_reg = 0
        num_foreground_cls = 0
        bbox_loss, cls_loss = [], []
        
        
        # Match original targets with default boxes
        for (targets_per_image, 
             bbox_regression_per_image, 
             cls_logits_per_image, 
             anchors_per_image, 
             matched_idxs_per_image
             ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            #matched_idxs_per_image
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground_reg += foreground_idxs_per_image.numel()

            
            # Compute regression loss
            foreground_matched_idxs_per_image = foreground_matched_idxs_per_image.to('cuda')

            matched_gt_boxes_per_image = targets_per_image["boxes"].to("cuda")[foreground_matched_idxs_per_image]

            #matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            
            bbox_loss.append(
                nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )
            
            ## Compute classification loss (focal loss)
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground_cls += foreground_idxs_per_image.sum()
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            


            labels = targets_per_image["labels"].to("cuda")
            
            gt_classes_target[
                foreground_idxs_per_image,
                labels[foreground_matched_idxs_per_image],
            ] = 1.0
            

            cls_loss.append(
                torchvision.ops.focal_loss.sigmoid_focal_loss(
                    cls_logits_per_image,
                    gt_classes_target,
                    reduction="sum",
                )
            )

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)
        
        del labels, gt_classes_target, foreground_matched_idxs_per_image, foreground_idxs_per_image, targets
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "bbox_regression": bbox_loss.sum() / max(1, num_foreground_reg),
            "classification": cls_loss.sum() / max(1, num_foreground_cls),
        }

    def on_train_epoch_start(self):
        self.train_detections, self.train_targets = [], []

    def on_validation_epoch_start(self):
        self.val_detections, self.val_targets = [], []
        
    def on_test_epoch_start(self):
        self.test_detections, self.test_targets = [], []

    def on_train_epoch_end(self):
        self.on_mode_epoch_end(mode="train")
        
    def on_validation_epoch_end(self):
        self.on_mode_epoch_end(mode="val")
        
    def on_test_epoch_end(self):
        self.on_mode_epoch_end(mode="test")

    def step(self, batch, batch_idx, mode):
        events, targets = batch['inputs'], batch['labels']
        frame_is_labeled = batch['frame_is_labeled']
        #forward pass
        features, head_outputs = self(events)

        # Anchors generation
        anchors = self.anchor_generator(features, (self.height, self.width))

        # TARGET making
        
        features, targets = self.prepare_features_targets(features=features, targets=targets, frame_is_labeled=frame_is_labeled)
        # match targets with anchors
        matched_idxs = []
 
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0 or targets_per_image["labels"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue
                
            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"].to("cuda"), anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        # Loss computation
        loss = None
        if mode != "test":
            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)

            bbox_loss = losses['bbox_regression']
            cls_loss = losses['classification']

            self.log(f'{mode}_loss_bbox', bbox_loss, on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)
            self.log(f'{mode}_loss_classif', cls_loss, on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)

            loss = bbox_loss + cls_loss
            self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            
        # Postprocessing for mAP computation
        if mode != "train":
            detections = self.postprocess_detections(head_outputs, anchors)
            if mode == "test":
                detections = list(map(filter_boxes, detections))
                targets = list(map(filter_boxes, targets))

            getattr(self, f"{mode}_detections").extend([{k: v.cpu().detach() for k,v in d.items()} for d in detections])
            getattr(self, f"{mode}_targets").extend([{k: v.cpu().detach() for k,v in t.items()} for t in targets])

        spikingjelly.activation_based.functional.reset_net(self.backbone)
        del losses, bbox_loss, cls_loss
        gc.collect()
        torch.cuda.empty_cache()
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")
    
    def on_mode_epoch_end(self, mode):
        if mode != "train":
            print(f"[{self.current_epoch}] {mode} results:")
            
            targets = getattr(self, f"{mode}_targets")

            detections = getattr(self, f"{mode}_detections")
            if detections == []:
                print("No detections")
                return
            
            h, w = self.height, self.width
            
            stats = coco_eval(
                targets, 
                detections, 
                height=h, width=w, 
                labelmap=("pedestrian", "two wheeler", "car")
                )

            keys = [
                'val_AP_IoU=.5:.05:.95', 'val_AP_IoU=.5', 'val_AP_IoU=.75', 
                'val_AP_small', 'val_AP_medium', 'val_AP_large',
                'val_AR_det=1', 'val_AR_det=10', 'val_AR_det=100',
                'val_AR_small', 'val_AR_medium', 'val_AR_large',
            ]
            
            stats_dict = {k:v for k,v in zip(keys, stats)}
            self.log_dict(stats_dict, sync_dist=True)
            del stats_dict, detections, targets
            gc.collect()
            torch.cuda.empty_cache()

    def postprocess_detections(
        self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_logits = head_outputs["cls_logits"]
                                   
        detections = []

        for boxes, logits, anchors in zip(bbox_regression, pred_logits, image_anchors):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, self.args.image_shape)

            image_boxes, image_scores, image_labels = [], [], []
            for label in range(len(self.label_map)):
                logits_per_class = logits[:, label]
                score = torch.sigmoid(logits_per_class).flatten()
                
                # remove low scoring boxes
                keep_idxs = score > self.args.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.args.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64))

            image_boxes = torch.cat(image_boxes, dim=0).type(torch.float16)
            image_scores = torch.cat(image_scores, dim=0).type(torch.float16)
            image_labels = torch.cat(image_labels, dim=0)#.type(torch.float16)

            #non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.args.nms_thresh)
            keep = keep[: self.args.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        del bbox_regression, pred_logits, image_anchors, image_scores, image_labels, image_boxes
        gc.collect()
        torch.cuda.empty_cache()
        return detections

    def configure_optimizers(self):
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Number of parameters:', n_parameters)
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.args.wd,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            self.args.epochs,
            eta_min=1e-5
        )
        return [optimizer], [scheduler]
