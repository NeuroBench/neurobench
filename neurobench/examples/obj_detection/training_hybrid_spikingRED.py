import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from tqdm import tqdm
from itertools import chain
from collections import defaultdict

import numpy as np
np.set_printoptions(precision=16,threshold=np.inf)
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from metavision_ml.data import box_processing as box_api
from metavision_ml.detection.losses import DetectionLoss
from metavision_ml.metrics.coco_eval import CocoEvaluator
from metavision_sdk_core import EventBbox

from obj_det_model import  Vanilla_lif
from training_utils import seq_dataloader

from metavision_ml.detection.anchors import Anchors
from metavision_ml.detection.rpn import BoxHead

torch.manual_seed(0)
np.random.seed(0)
scaler = torch.cuda.amp.GradScaler()


def bboxes_to_box_vectors(bbox):
    if isinstance(bbox, list):
        return [bboxes_to_box_vectors(item) for item in bbox]
    elif isinstance(bbox, np.ndarray) and bbox.dtype != np.float32:
        return box_api.bboxes_to_box_vectors(bbox)
    else:
        return bbox
    
class Trainer:

    def __init__(self, model: nn.Module, BoxHead,box_coder, dataloader, log_dir):
        self.device = 'cuda' #if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.pd = BoxHead.to(self.device)
        self.box_coder = box_coder
        self.seq_dataloader_train = self.dataloader.seq_dataloader_train
        self.seq_dataloader_val = self.dataloader.seq_dataloader_val
        self.seq_dataloader_test = self.dataloader.seq_dataloader_test
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                           {'params': self.pd.parameters()}
                                           ],
                                           lr=0.0002, weight_decay=1e-5)
       
        self.criterion = DetectionLoss("softmax_focal_loss")
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,  gamma=0.98)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr = 0.0002, epochs = 20, steps_per_epoch =1)
        
        self.cnt = 0
        self.cnt_train = 0
        self.cnt_val = 0
        self.label_map = ['background'] + self.dataloader.wanted_keys
        self.logger = SummaryWriter(log_dir=log_dir)

        # self.model.load_state_dict(torch.load('/home/shenqi/Master_thesis/test/my_training/3label_rawdataset/25_model.pth',map_location=torch.device(self.device)))
        # self.pd.load_state_dict(torch.load('/home/shenqi/Master_thesis/test/my_training/3label_rawdataset/25_pd.pth',map_location=torch.device(self.device)))

    def accumulate_predictions(self, preds, targets, video_infos, frame_is_labeled):
       
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

                # if ts < self.hparams.skip_us:
                #     continue

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

    def inference_epoch_end(self, outputs):
      
        print('==> Start evaluation')
        dt_detections = defaultdict(list)
        gt_detections = defaultdict(list)

        for item in outputs:
            for k, v in item['gt'].items():
                gt_detections[k].extend(v)
            for k, v in item['dt'].items():
                dt_detections[k].extend(v)

        evaluator = CocoEvaluator(classes=self.label_map, height=self.dataloader.height, width=self.dataloader.width)
        for key in gt_detections:
            evaluator.partial_eval([np.concatenate(gt_detections[key])], [np.concatenate(dt_detections[key])])
        coco_kpi = evaluator.accumulate()
        return coco_kpi
    
    def select_valid_frames(self, xs, targets, frame_is_labeled):
        frame_is_labeled = frame_is_labeled.bool()
        mask = frame_is_labeled.view(-1)
        xs = [item[mask] for item in xs]
        targets = [
            [targets[r][c] for c in range(len(frame_is_labeled[0])) if frame_is_labeled[r][c]]
            for r in range(len(frame_is_labeled))]
        return xs, targets
    
    def compute_loss(self, data):
        inputs = data['inputs'].to(device=self.device)
        targets = bboxes_to_box_vectors(data['labels'])
        frame_is_labeled = data['frame_is_labeled']
        with torch.cuda.amp.autocast():
            self.model.reset(data['mask_keep_memory'])
            xs = self.model(inputs)
            if frame_is_labeled.sum().item() == 0:
                return None
            xs, targets = self.select_valid_frames(xs, targets, frame_is_labeled)
            loc_preds, cls_preds = self.pd(xs)
            targets = list(chain.from_iterable(targets))
            targets = self.box_coder.encode(xs, inputs, targets)
            assert targets['cls'].shape[1] == cls_preds.shape[1]
            loss_dict = self.criterion(loc_preds, targets['loc'], cls_preds, targets["cls"])
        return loss_dict
    
    def compute_loss_and_inference(self, data):
        inputs = data['inputs'].to(device=self.device)
        targets = bboxes_to_box_vectors(data['labels'])
        
        frame_is_labeled = data['frame_is_labeled']
        with torch.no_grad():
            xs = self.model(inputs)
            xs_val = xs
            loc_preds_val, cls_preds_val = self.pd(xs_val)

            if frame_is_labeled.sum().item() == 0:
                return None
            xs, targets = self.select_valid_frames(xs, targets, frame_is_labeled)
            loc_preds, cls_preds = self.pd(xs)
            
            score_thresh=0.05
            nms_thresh=0.5
            max_boxes_per_input = 500
            scores = self.pd.get_scores(cls_preds_val)
            scores = scores.to('cpu')

            targets = list(chain.from_iterable(targets))
            targets = self.box_coder.encode(xs, inputs, targets)
            assert targets['cls'].shape[1] == cls_preds.shape[1]
            loss_dict = self.criterion(loc_preds, targets['loc'], cls_preds, targets["cls"])
            
            
            for i, feat in enumerate(xs_val):
                xs_val[i] = xs_val[i].to('cpu')
            inputs = data['inputs'].to('cpu')
            loc_preds_val = loc_preds_val.to('cpu')
            preds = self.box_coder.decode(xs_val, inputs, loc_preds_val, scores, batch_size=inputs.shape[1], score_thresh=score_thresh,
                                     nms_thresh=nms_thresh, max_boxes_per_input=max_boxes_per_input)
            dt_dic, gt_dic = self.accumulate_predictions(
                preds,
                data["labels"],
                data["video_infos"],
                data["frame_is_labeled"])

        return loss_dict, {'dt': dt_dic, 'gt': gt_dic}  

    def train_epoch(self, seq_dataloader_train):
        
        seq_dataloader_train.dataset.shuffle()

        self.model.train()
        self.pd.train()
        epoch_metrics = {
            "loss": [],
            'loc_loss':[],
            'cls_loss':[]
        }

        sys.stdout.flush()
        
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        with tqdm(total=len(seq_dataloader_train), desc=f'Training',ncols=120) as pbar:
                        
            for data in seq_dataloader_train:
                sys.stdout.flush()
                self.cnt_train += 1

                loss_dict = self.compute_loss(data)
                if loss_dict is None:
                    continue
                
                loss = sum([value for key, value in loss_dict.items()])
                if torch.isnan(loss):
                    print(data["video_infos"])
                    raise Exception

                self.optimizer.zero_grad()
                scaler.scale(loss).backward(retain_graph=True)
                scaler.step(self.optimizer)
                scaler.update()
                
                step_metrics = {
                    'loss': loss.item(),
                    'loc_loss': loss_dict['loc_loss'].item(),
                    'cls_loss': loss_dict['cls_loss'].item()
                }

                pbar.set_postfix(**step_metrics)
                pbar.update(1)

                for k,v in step_metrics.items():
                    epoch_metrics[k].append(v)
                self.logger.add_scalar('train loss', loss.item(),self.cnt_train)
                self.logger.add_scalar('cls loss', loss_dict['cls_loss'].item(),self.cnt_train )
                self.logger.add_scalar('loc loss', loss_dict['loc_loss'].item(),self.cnt_train )

        return epoch_metrics
    
    def val_epoch(self, seq_dataloader_val_or_test, epochs):
        seq_dataloader_val_or_test.dataset.shuffle()
        self.model.eval()
        self.pd.eval()
        output_val_list = []
        loss_val_mean = 0
        self.cnt_val = 0

        with tqdm(total=len(seq_dataloader_val_or_test), desc=f'Validation',ncols=120) as pbar:
                        
            for data in seq_dataloader_val_or_test:
                sys.stdout.flush()
                
                data['inputs'] = data['inputs'].to(device=self.device)
                
                loss_dict_val, output_val = self.compute_loss_and_inference(data)        
                output_val_list.append(output_val) 

                if loss_dict_val is None:
                    print("loss is none")
                    continue
                loss_val = sum([value for key, value in loss_dict_val.items()])
                loss_val_mean += loss_val
                self.cnt_val += 1
                pbar.update(1)
                
            self.logger.add_scalar('val loss', loss_val_mean/self.cnt_val, epochs)
            coco_val_result = self.inference_epoch_end(output_val_list)
            print(coco_val_result)
            return coco_val_result
                
    
    def fit(self, epochs: int):
        dl_train = self.seq_dataloader_train
        dl_val = self.seq_dataloader_val
        dl_test = self.seq_dataloader_test
        
        
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}')
            
            self.train_epoch(dl_train)
            
            path_model = './save_models/' + str(epoch) + '_model.pth'
            path_pd = './save_models/' + str(epoch) + '_pd.pth'
            torch.save(self.model.state_dict(),path_model)
            torch.save(self.pd.state_dict(),path_pd)

            
            metrics_val = self.val_epoch(dl_val,epoch)
            # metrics_test = self.val_epoch(dl_test)

            with open("result.txt",'a') as f:
                f.write(str(epoch))
                f.write(str(metrics_val))
                # f.write('test:')
                # f.write(str(metrics_test))
                f.write('\n')
                 
            self.scheduler.step() 







dataloader = seq_dataloader()
model = Vanilla_lif(cin = dataloader.in_channels, cout = 256, base = 16)
box_coder = Anchors(num_levels=model.levels, anchor_list="PSEE_ANCHORS", variances=[0.1, 0.2])
head = BoxHead(model.cout, box_coder.num_anchors, len(dataloader.wanted_keys) + 1, 0)

trainer = Trainer(model, head, box_coder,dataloader,log_dir='./log')
trainer.fit(epochs=20)