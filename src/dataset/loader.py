"""Data loader."""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.utils.data.sampler as sampler
from torchvision import datasets
from torchvision import transforms

from core.config import cfg
import dataset.paths as dp
from dataset.cifar10 import Cifar10


# Supported datasets
_DATASET_CATALOG = {"cifar10": Cifar10}

_DEFAULT_TRANSFORM = transforms.ToTensor()


def load_and_prepare_data():
    "Method to load dataset, perform whitening, etc"
    pass


def _construct_loader(
    dataset_name, root, split, batch_size, shuffle, drop_last, transform
):
    """Constructs the data loader for the given dataset."""
    assert dataset_name in _DATASET_CATALOG.keys(), f"Dataset '{dataset_name}' not supported"
    assert split in [
        "train",
        "valid",
        "test",
    ], "split can only be 'train', 'test' or 'valid'"

    train = True if split == "train" else False
    # Retrieve the data path for the dataset
    data_path = dp.get_data_path(dataset_name)
    # Construct the dataset
    dataset = _DATASET_CATALOG[dataset_name](
        datapath=root, split=split
    )
    ds_sampler = None
    if split != "test":
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split_ids = int(np.floor(cfg.TRAIN.VALID_SIZE * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, valid_indices = indices[split_ids:], indices[:split_ids]
        if split == "train":
            ds_sampler = sampler.SubsetRandomSampler(train_indices)
        else:
            ds_sampler = sampler.SubsetRandomSampler(valid_indices)

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=ds_sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(root, transform=_DEFAULT_TRANSFORM):
    """Train loader wrapper."""
    return _construct_loader(
        root=root,
        dataset_name=cfg.TRAIN.DATASET,
        split="train",
        transform=transform,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )


def construct_val_loader(root, transform=_DEFAULT_TRANSFORM):
    """Val loader wrapper."""
    return _construct_loader(
        root=root,
        dataset_name=cfg.TRAIN.DATASET,
        split="valid",
        transform=transform,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )


def construct_test_loader(root, transform=_DEFAULT_TRANSFORM):
    """Test loader wrapper."""
    return _construct_loader(
        root=root,
        dataset_name=cfg.TEST.DATASET,
        split="test",
        transform=transform,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
