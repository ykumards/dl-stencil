import argparse
from datetime import datetime
from pathlib import Path
import os
import sys
from typing import List
from tqdm.autonotebook import tqdm
import warnings

import numpy as np

import torch
import torch.nn as nn

from core.config import assert_and_infer_cfg, cfg, dump_cfg
import core.model_builder as model_builder
import core.losses as losses
import core.optimizers as optim
import dataset.loader as loader
import utils.checkpoint as cu
import utils.common as common
import utils.logging as lu
import utils.metrics as mu
import utils.net as nu
from utils.meters import Meter
from utils.tb_logger import TensorboardLogger


logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train MNIST model")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See src/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def log_model_info(model):
    """Logs model info"""
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(mu.params_count(model)))
    logger.info("Acts: {:,}".format(mu.acts_count(model)))
    logger.info("Flops: {:,}".format(mu.flops_count(model)))


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or (
        cur_epoch + 1
    ) == cfg.OPTIM.MAX_EPOCH


def _prepare_batch(batch):
    x = batch[0].to(cfg.DEVICE)
    return x


def train_epoch(
    train_loader: torch.data.utils.DataLoader,
    model: nn.Module,
    loss_funs: List,
    optimizer: torch.optim.Optimizer,
    train_meter: Meter,
    cur_epoch: int,
    mode="train",
    tb=None,
):
    """Performs one epoch of training."""
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for cur_iter, batch in pbar:
        # Transfer the data to the current GPU device
        x = _prepare_batch(batch)
        # Perform the forward pass
        x_pred, mu, logvar = model(x)
        # Compute reconstruction loss
        reconc_loss = loss_funs[0](x_pred, x)
        # Compute kld loss
        kl_loss = loss_funs[1](mu, logvar)
        # Total loss is sum of reconc and kl_div
        loss = reconc_loss + kl_loss
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIM.GRAD_CLIP_T)
        # Update the parameters
        optimizer.step()
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        reconc_loss, kl_loss = reconc_loss.item(), kl_loss.item()

        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(loss, x.size(0), lr)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.print_epoch_stats(cur_epoch)

    if tb is not None:
        # log scalars
        epoch_stats = train_meter.get_epoch_stats(cur_epoch)
        tb.write_scalar(
            epoch_stats,
            cur_epoch,
            ["loss", "kl_loss", "reconc_loss", "label_err"],
            tag=mode,
        )

        tb.write_optim_scalar(optimizer, cur_epoch, tag=mode)
        tb.write_model_weight_hist(model, cur_epoch, tag=mode)
        tb.write_model_grad_hist(model, cur_epoch, tag=mode)

    train_meter.reset()


@torch.no_grad()
def test_epoch(
    test_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_funs: List,
    test_meter: Mete,
    cur_epoch: int,
    mode="test",
    tb=None,
):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
    for cur_iter, batch in pbar:
        # Transfer the data to the current GPU device
        x = _prepare_batch(batch)
        # Perform the forward pass
        x_pred, mu, logvar = model(x)
        # Compute reconstruction loss
        reconc_loss = loss_funs[0](x_pred, x)
        # Compute kld loss
        kl_loss = loss_funs[1](mu, logvar)
        # Total loss is sum of reconc and kl_div
        loss = reconc_loss + kl_loss
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        reconc_loss, kl_loss = reconc_loss.item(), kl_loss.item()

        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(loss, x.size(0))
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.print_epoch_stats(cur_epoch)
    epoch_loss = test_meter.loss_total / test_meter.num_samples

    if tb is not None:
        # log scalars
        epoch_stats = test_meter.get_epoch_stats(cur_epoch)
        tb.write_scalar(
            epoch_stats,
            cur_epoch,
            ["loss", "kl_loss", "reconc_loss", "label_err"],
            tag=mode,
        )

    test_meter.reset()
    return epoch_loss


def train_model():
    """Trains the model."""
    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model()
    log_model_info(model)

    # Define the loss function
    loss_funs = losses.get_loss_fun_vae()
    # Construct the optimizer
    optimizer = optim.construct_optimizer(model)

    start_epoch = 0
    min_val_loss = np.inf
    cur_patience = 0

    # Create data loaders
    # train_data, val_data, test_data = loader.load_and_prepare_data()
    train_loader = loader.construct_train_loader(root=cfg.PATHS.DATAPATH)
    val_loader = loader.construct_val_loader(root=cfg.PATHS.DATAPATH)
    test_loader = loader.construct_test_loader(root=cfg.PATHS.DATAPATH)

    # Create meters
    train_meter = Meter(len(train_loader), cfg.TRAIN.BATCH_SIZE, mode="train")
    val_meter = Meter(len(val_loader), cfg.TEST.BATCH_SIZE, mode="valid")
    test_meter = Meter(len(val_loader), cfg.TEST.BATCH_SIZE, mode="test")

    # setup tb logging
    tb = None
    if cfg.IS_TB_LOG:
        tb = TensorboardLogger(log_dir=cfg.PATHS.TB_OUT_DIR, flush_secs=30)

    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(
            train_loader,
            model,
            loss_funs
            optimizer,
            train_meter,
            cur_epoch,
            mode="train",
            tb=tb,
        )
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if cu.is_checkpoint_epoch(cur_epoch):
            checkpoint_file = cu.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        if is_eval_epoch(cur_epoch):
            val_loss = test_epoch(
                val_loader, model, loss_funs val_meter, cur_epoch, mode="valid", tb=tb
            )
            # Save the best model based on val score
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                cur_patience = 0
                checkpoint_file = cu.save_best_loss_checkpoint(
                    model, optimizer, cur_epoch, val_loss
                )
                print(f"Wrote best score checkpoint to: {checkpoint_file}")
            # Handle early stopping based on val score
            elif val_loss - cfg.TRAIN.ES_THRESHOLD > min_val_loss:
                cur_patience += 1
                print(
                    f"Val loss larger than min value, patience at: {cur_patience} (max {cfg.TRAIN.ES_PATIENCE})"
                )
                if cur_patience > cfg.TRAIN.ES_PATIENCE:
                    logger.info(f"ES patience hit at {cur_epoch} epochs, quitting")
                    break

    best_checkpoint = cu.get_best_score_checkpoint()
    best_epoch = cu.load_checkpoint(best_checkpoint, model, optimizer)
    print(f"Loaded checkpoint from epoch: {best_epoch+1}")

    print("=" * 100)
    test_epoch(
        train_loader, model, loss_funs train_meter, cur_epoch, mode="train", tb=None
    )
    test_epoch(
        test_loader, model, loss_funs test_meter, cur_epoch, mode="test", tb=None
    )

    if tb is not None:
        tb.close()


def run():
    lu.setup_logging()
    # Show the config
    logger.info("Config:\n{}".format(cfg))
    # seed everything for reproducability
    common.seed_everything(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # train the model
    train_model()

    # TODO test the model
    # test_model()


def main():
    # Parse cmd line args
    args = parse_args()
    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_and_infer_cfg()
    cfg.PATHS.OUT_DIR = os.path.join(cfg.PATHS.OUT_DIR, cfg.PATHS.EXPERIMENT_NAME)
    cfg.PATHS.TB_OUT_DIR = os.path.join(
        cfg.PATHS.OUT_DIR, "tb_logs", cfg.PATHS.TIMESTAMP
    )
    cfg.PATHS.MODEL_OUT_DIR = os.path.join(
        cfg.PATHS.OUT_DIR, "saved_models", cfg.PATHS.TIMESTAMP
    )

    cfg.freeze()

    # Ensure that the output dir exists
    try:
        os.makedirs(cfg.PATHS.OUT_DIR, exist_ok=True)
        os.makedirs(cfg.PATHS.MODEL_OUT_DIR, exist_ok=False)
        os.makedirs(cfg.PATHS.TB_OUT_DIR, exist_ok=False)
    except FileExistsError:
        print("Wait for a minute and try again :)")
        exit()

    if cfg.TUNE_LR:
        pass

    # Save the config
    dump_cfg()

    # let's gooo
    run()


if __name__ == "__main__":
    main()
