from os.path import join
import sys
import argparse

try:
    import comet_ml
except ImportError:
    print("Comet is not installed, Comet logger will not be available.")

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from datasets.classification_datasets import NCARSClassificationDataset, GEN1ClassificationDataset
from models.utils import get_model
from classification_module import ClassificationLitModule



def main():
    parser = argparse.ArgumentParser(description='Classify event dataset')
    parser.add_argument('-device', default=0, type=int, help='device')
    parser.add_argument('-precision', default=16, type=int, help='whether to use AMP {16, 32, 64}')

    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-sample_size', default=100000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=2, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(304,240), type=tuple, help='spatial resolution of events')

    parser.add_argument('-dataset', default='ncars', type=str, help='dataset used {NCARS, GEN1}')
    parser.add_argument('-path', default='PropheseeNCARS', type=str, help='dataset used. {NCARS, GEN1}')
    parser.add_argument('-undersample_cars_percent', default='0.24', type=float, help=
                        'Undersample cars in Prophesse GEN1 Classification by using only x percent of cars.')

    parser.add_argument('-model', default='vgg-11', type=str, help='model used {squeezenet-v, vgg-v, mobilenet-v, densenet-v}')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-lr', default=5e-3, type=float, help='learning rate used')
    parser.add_argument('-epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-test', action='store_true', help='whether to test the model')

    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-comet_api', default=None, type=str, help='api key for Comet Logger')

    args = parser.parse_args()
    print(args)

    if args.dataset == "ncars":
        dataset = NCARSClassificationDataset
    elif args.dataset == "gen1":
        dataset = GEN1ClassificationDataset
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")

    train_dataset = dataset(args, mode="train")
    test_dataset = dataset(args, mode="test")

    train_dataloader = DataLoader(train_dataset, batch_size=args.b, num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.b, num_workers=8)

    model = get_model(args)
    module = ClassificationLitModule(model, epochs=args.epochs, lr=args.lr)

    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
        ckpt_path = join("pretrained", join(args.model, args.pretrained))
        module = module.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)

    callbacks=[]
    if args.save_ckpt:
        ckpt_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=f"ckpt-{args.dataset}-{args.model}/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{val_acc:.4f}",
            save_top_k=3,
            mode='max',
        )
        callbacks.append(ckpt_callback)

    logger = None
    if args.comet_api:
        try:
            comet_logger = CometLogger(
                api_key=args.comet_api,
                project_name=f"classif-{args.dataset}-{args.model}/",
                save_dir="comet_logs",
                log_code=True,
            )
            logger = comet_logger
        except ImportError:
            print("Comet is not installed, Comet logger will not be available.")

    trainer = pl.Trainer(
        gpus=[args.device], gradient_clip_val=1., max_epochs=args.epochs,
        limit_train_batches=1., limit_val_batches=1.,
        check_val_every_n_epoch=1,
        deterministic=False,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
    )

    if args.train:
        trainer.fit(module, train_dataloader, test_dataloader)
    if args.test:
        trainer.test(module, test_dataloader)

if __name__ == '__main__':
    main()
