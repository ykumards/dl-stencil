"""Functions that handle saving and loading of checkpoints."""

import os
import numpy as np

import torch
from core.config import cfg


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"
_SCORE_NAME_PREFIX = "best_model_score="
# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return cfg.PATHS.MODEL_OUT_DIR


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_score_checkpoint(loss_score):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:4f}.pyth".format(_SCORE_NAME_PREFIX, loss_score)
    return os.path.join(get_checkpoint_dir(), name)


def get_best_score_checkpoint():
    """Retrieves the checkpoint with lowest loss score."""
    # breakpoint()
    checkpoint_dir = get_checkpoint_dir()
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if "best_" in f]
    best_checkpoint_val_loss = [
        float(
            ".".join(x.split("=")[1].split(".")[0:2])
        )  # checkpoint in format: name-00.00.pyth
        for x in checkpoints
    ]
    best_idx = np.array(best_checkpoint_val_loss).argmin()
    name = checkpoints[best_idx]
    return os.path.join(get_checkpoint_dir(), name)


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def is_checkpoint_epoch(cur_epoch):
    """Determines if a checkpoint should be saved on current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_best_loss_checkpoint(model, optimizer, epoch, loss_score):
    "Saves the best model based on loss score."
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.state_dict()
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_score_checkpoint(loss_score)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def save_checkpoint(model, optimizer, epoch):
    """Saves a checkpoint."""
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.state_dict()
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if cfg.NUM_GPUS > 1 else model
    ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]


def load_best_score_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if cfg.NUM_GPUS > 1 else model
    ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]
