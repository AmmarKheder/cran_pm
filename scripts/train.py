#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cranpm.training.trainer import CranPMLightning
from cranpm.data.dataset import CranPMDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tc = config["train"]
    datamodule = CranPMDataModule(config)
    model = CranPMLightning(config)

    if args.finetune:
        import torch
        ckpt = torch.load(args.finetune, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    ckpt_dir = os.environ.get("CKPT_DIR", "checkpoints")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="cranpm-{epoch:03d}",
        monitor="val/rmse",
        auto_insert_metric_name=False,
        mode="min",
        save_top_k=tc.get("save_top_k", 3),
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/rmse",
        patience=tc.get("early_stopping_patience", 100),
        mode="min",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(save_dir="logs", name="cranpm", default_hp_metric=False)

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    strategy = (
        "ddp" if num_nodes > 1 or int(os.environ.get("SLURM_GPUS_ON_NODE", 1)) > 1
        else "auto"
    )

    trainer_kwargs = dict(
        max_epochs=tc.get("epochs", 300),
        max_steps=tc.get("max_steps", -1),
        precision=tc.get("precision", "bf16-mixed"),
        gradient_clip_val=tc.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=tc.get("accumulate_grad_batches", 1),
        log_every_n_steps=tc.get("log_every_n_steps", 10),
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=logger,
        strategy=strategy,
        num_nodes=num_nodes,
        devices="auto",
        accelerator="auto",
        enable_progress_bar=True,
    )
    if "check_val_every_n_epoch" in tc:
        trainer_kwargs["check_val_every_n_epoch"] = tc["check_val_every_n_epoch"]
    elif "val_check_interval" in tc:
        trainer_kwargs["val_check_interval"] = tc["val_check_interval"]

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
