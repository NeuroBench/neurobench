from os.path import join
import sys
import argparse

# comet_available = False
try:
    import comet_ml
    # comet_available = True
except ImportError:
    print("Comet is not installed, Comet logger will not be available.")

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from datasets.gen1_od_dataset import GEN1DetectionDataset
from datasets.gen2_dataset import GEN2DetectionDataset
from object_detection_module import DetectionLitModule

def collate_fn(batch):
    samples = [item[0] for item in batch]
    samples = torch.stack(samples, 0)
    
    targets = [item[1] for item in batch]
    return [samples, targets]

def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    # Dataset
    parser.add_argument('-dataset', default='gen2', type=str, help='dataset used {GEN1}')
    parser.add_argument('-path', default='/Data/pgi-15/datasets/propheseeMini/mini_dataset', type=str, help='path to dataset location')
    parser.add_argument('-num_classes', default=7, type=int, help='number of classes')

    # Data
    parser.add_argument('-b', default=2, type=int, help='batch size')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(720,1280), type=tuple, help='spatial resolution of events')

    # Training
    parser.add_argument('-epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate used')
    parser.add_argument('-wd', default=1e-4, type=float, help='weight decay used')
    parser.add_argument('-num_workers', default=0, type=int, help='number of workers for dataloaders')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')
    parser.add_argument('-device', default=0, type=int, help='device')
    parser.add_argument('-precision', default=16, type=int, help='whether to use AMP {16, 32, 64}')
    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    # parser.add_argument('-comet_api', default=None, type=str, help='api key for Comet Logger')
    parser.add_argument('-comet_api', default='CqwgdCAF77gOa2bRpOouLMW2U', type=str, help='api key for Comet Logger')
    # Backbone
    parser.add_argument('-backbone', default='vgg-11', type=str, help='model used {squeezenet-v, vgg-v, mobilenet-v, densenet-v}', dest='model')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained_backbone', default=None, type=str, help='path to pretrained backbone model')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-extras', type=int, default=[640, 320, 320], nargs=4, help='number of channels for extra layers after the backbone')


    # Priors
    parser.add_argument('-min_ratio', default=0.05, type=float, help='min ratio for priors\' box generation')
    parser.add_argument('-max_ratio', default=0.80, type=float, help='max ratio for priors\' box generation')
    parser.add_argument('-aspect_ratios', default=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], type=int, help='aspect ratios for priors\' box generation')

    # Loss parameters
    parser.add_argument('-box_coder_weights', default=[10.0, 10.0, 5.0, 5.0], type=float, nargs=4, help='weights for the BoxCoder class')
    parser.add_argument('-iou_threshold', default=0.50, type=float, help='intersection over union threshold for the SSDMatcher class')
    parser.add_argument('-score_thresh', default=0.01, type=float, help='score threshold used for postprocessing the detections')
    parser.add_argument('-nms_thresh', default=0.45, type=float, help='NMS threshold used for postprocessing the detections')
    parser.add_argument('-topk_candidates', default=200, type=int, help='number of best detections to keep before NMS')
    parser.add_argument('-detections_per_img', default=100, type=int, help='number of best detections to keep after NMS')

    args = parser.parse_args()
    print(args)

    if args.dataset == "gen1":
        dataset = GEN1DetectionDataset

    if args.dataset == "gen2":
        dataset = GEN2DetectionDataset
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    module = DetectionLitModule(args)

    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
        ckpt_path = join("pretrained", join(args.model, args.pretrained))
        module = module.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)

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
    if args.comet_api:
        try:
            comet_logger = CometLogger(
                api_key=args.comet_api,
                project_name=f"od-{args.dataset}-{args.model}/",
                save_dir="comet_logs",
                log_code=True,
            )
            logger = comet_logger
        except ImportError:
            print("Comet is not installed, Comet logger will not be available.")
            

    trainer = pl.Trainer(
        gpus=[args.device], gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=1., limit_val_batches=.25,
        check_val_every_n_epoch=5,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
    )

    if args.train:
        train_dataset = dataset(args, mode="train")
        val_dataset = dataset(args, mode="val")    
        train_dataloader = DataLoader(train_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers)
        
        trainer.fit(module, train_dataloader, val_dataloader)
    if args.test:
        test_dataset = dataset(args, mode="test")
        test_dataloader = DataLoader(test_dataset, batch_size=args.b, collate_fn=collate_fn, num_workers=args.num_workers)

        trainer.test(module, test_dataloader)

if __name__ == '__main__':
    main()
