import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
# this gives us the abs path of benchmarks folder
ROOT_DIR = str(cur_dir.parent.parent.absolute())


@dataclass
class DataParams:
    t_delta: int = 50000
    channels: int = 6  # histograms have two channels
    num_tbins: int = 12
    max_incr_per_pixel: int = 5
    start_ts: int = 0
    height: int = 360
    width: int = 640
    batch_size: int = 2
    max_duration: int = None
    preprocess_fn: str = "multi_channel_timesurface"
    n_processes: int = 16
    classes_to_use: Tuple = ('pedestrian', 'two wheeler', 'car')


@dataclass
class TrainingParams:
    n_gpus: int = 4
    epochs: int = 20
    lr: int = 1e-3
    weight_decay: int = 1e-4
    backbone_model: str = 'vgg-11'
    min_max_ratio: Tuple = (0.05, 0.80)
    iou_threshold: int = 0.5
    score_threshold: int = 0.03
    nms_threshold: int = 0.7
    topk_candidates: int = 100
    detections_per_img: int = 50
