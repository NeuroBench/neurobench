import argparse

import pytorch_lightning as pl
from model import LightningDetectionModel
from dataset import Gen4DetectionDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_tbins", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler_step_gamma",
        type=float,
        default=None,
        help="learning rate scheduler step param (disabled if None)",
    )
    parser.add_argument("--dataset_path", type=str, default="data/Gen 4 Histograms")
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(123)

    class_selection = ["pedestrian", "two wheeler", "car"]
    dataset = Gen4DetectionDataset(
        batch_size=args.batch_size,
        num_tbins=args.num_tbins,
        class_selection=class_selection,
        dataset_path=args.dataset_path,
    )

    model = LightningDetectionModel(
        batch_size=args.batch_size,
        feature_extractor="Vanilla",
        in_channels=dataset.channels,
        feature_channels_out=256,
        feature_base=16,
        classes=class_selection,
        anchor_list="PSEE_ANCHORS",
        max_boxes_per_input=500,
        lr=args.lr,
        lr_scheduler_step_gamma=args.lr_scheduler_step_gamma,
        height=dataset.height,
        width=dataset.width,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        filename="step={step}-epoch={epoch}-valid_loss={loss/valid}-valid_acc={accuracy/valid}",
        auto_insert_metric_name=False,
        save_top_k=5,
        mode="max",
    )

    logger = pl.loggers.TensorBoardLogger("./")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model, dataset)
    trainer.test(model, dataset)
