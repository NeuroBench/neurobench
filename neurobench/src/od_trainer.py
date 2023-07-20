from os.path import join
import argparse


import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from od_module import ObjDetectionModule
from data_utils import seq_dataloader
from utils import DataParams, TrainingParams

import logging
logging.getLogger().setLevel(logging.ERROR)


torch.manual_seed(42)

def main():
    data_params = DataParams()
    training_params = TrainingParams()

    parser = argparse.ArgumentParser(description='Classify event dataset')
    # Dataset
    parser.add_argument('-dataset', default='gen4', type=str, help='dataset used {GEN4}')
    #parser.add_argument('-path', default='/Data/p gi-15/datasets/propheseeMini/mini_dataset', type=str, help='path to dataset location')
    parser.add_argument('-num_classes', default=len(data_params.classes_to_use), type=int, help='number of classes')
    parser.add_argument('-target_classes', default=data_params.classes_to_use, type=int, help='number of classes')
    #parser.add_argument('-save_data', default=data_params.processed_data_path, type=str, help="Location of processed data")
    # Data
    parser.add_argument('-in_channels', default=data_params.channels, type=str, help="No of input channels")
    parser.add_argument('-b', default=data_params.batch_size, type=int, help='batch size')
    parser.add_argument('-sample_size', default=data_params.t_delta, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=data_params.num_tbins, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=3, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(data_params.width, data_params.height), type=tuple, help='spatial resolution of events')

    # Training
    parser.add_argument('-epochs', default=training_params.epochs, type=int, help='number of total epochs to run')
    parser.add_argument('-lr', default=training_params.lr, type=float, help='learning rate used')
    parser.add_argument('-wd', default=training_params.weight_decay, type=float, help='weight decay used')
    parser.add_argument('-num_workers', default=4, type=int, help='number of workers for dataloaders')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')
    parser.add_argument('-train', default=True, help='whether to test the model')
    #parser.add_argument('-device', default=0, type=int, help='device')
    parser.add_argument('-precision', default='16-mixed', type=str, help='whether to use AMP {16, 32, 64}')
    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    # Backbone
    parser.add_argument('-backbone', default='vgg-11', type=str, help='model used {squeezenet-v, vgg-v, mobilenet-v, densenet-v}', dest='model')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-extras', type=int, default=[320, 160, 160], nargs=4, help='number of channels for extra layers after the backbone')
    parser.add_argument('-tracking', default=False, type=str, help='Tracking or not')

    # Priors
    parser.add_argument('-min_ratio', default=training_params.min_max_ratio[0], type=float, help='min ratio for priors\' box generation')
    parser.add_argument('-max_ratio', default=training_params.min_max_ratio[1], type=float, help='max ratio for priors\' box generation')
    parser.add_argument('-aspect_ratios', default=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], type=int, help='aspect ratios for priors\' box generation')

    # Loss parameters
    parser.add_argument('-box_coder_weights', default=[10.0, 10.0, 5.0, 5.0], type=float, nargs=4, help='weights for the BoxCoder class')
    parser.add_argument('-iou_threshold', default=training_params.iou_threshold, type=float, help='intersection over union threshold for the SSDMatcher class')
    parser.add_argument('-score_thresh', default=training_params.score_threshold, type=float, help='score threshold used for postprocessing the detections')
    parser.add_argument('-nms_thresh', default=training_params.nms_threshold, type=float, help='NMS threshold used for postprocessing the detections')
    parser.add_argument('-topk_candidates', default=training_params.topk_candidates, type=int, help='number of best detections to keep before NMS')
    parser.add_argument('-detections_per_img', default=training_params.detections_per_img, type=int, help='number of best detections to keep after NMS')

    args = parser.parse_args()
    
    module = ObjDetectionModule(args)

    callbacks=[]
    if args.save_ckpt:
        ckpt_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=f"ckpt-od-{args.dataset}-{args.model}/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{train_loss:.4f}",
            save_top_k=3,
            mode='min',
        )
        callbacks.append(ckpt_callback)

    logger = None
    if args.tracking:
        try:
            wandblogger = WandbLogger(
                    project = "object_detection_benchmark",
                    notes = "Object Detection Prophese data: Part of benchmarking tasks for NMC hardware",
                    tags = ["od_prophese"],
                    #log_model=True

                )
            logger = wandblogger

        except ImportError:
            print("Wandb is not installed, logger will not be available.")
            

    trainer = pl.Trainer(
        gradient_clip_val=1., max_epochs=args.epochs,
        accelerator="gpu",
        devices=training_params.n_gpus,
        strategy="ddp_find_unused_parameters_true",
        limit_train_batches=1., 
        limit_val_batches=.25,
        check_val_every_n_epoch=4,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
    )
    loaders = seq_dataloader()
    if args.train:
        
        train_dataloader = loaders.seq_dataloader_train
        val_dataloader = loaders.seq_dataloader_val
        
        trainer.fit(module, train_dataloader, val_dataloader)
    if args.test:
        test_dataloader = loaders.seq_dataloader_test
        trainer.test(module, test_dataloader)

if __name__ == '__main__':
    main()
