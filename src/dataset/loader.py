"""Data loader."""

import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.config import cfg
import dataset.paths as dp
from dataset.twins import TwinsDataset


# Supported datasets
_DATASET_CATALOG = {"twins": TwinsDataset,}


def load_and_prepare_data():
    DATAPATH = Path('/data/counterfact_ds/')
    dump = np.load(DATAPATH/'twins_realized_dum.npz')
    data = dump.f.arr_0

    train_data, ts = train_test_split(data, test_size=0.3, random_state=cfg.RNG_SEED)
    test_data, val_data = train_test_split(ts, test_size=0.5, random_state=cfg.RNG_SEED)

    # Handle data scaling
    scaler = StandardScaler()
    # index 5:14 is for continuous features
    scaler.fit(train_data[:, 5:14])
    train_data[:, 5:14] = scaler.transform(train_data[:, 5:14])
    val_data[:, 5:14] = scaler.transform(val_data[:, 5:14])
    test_data[:, 5:14] = scaler.transform(test_data[:, 5:14])

    # 2 => y0, 3 => y1
    print(f"True ATE: {(data[:, 3] - data[:, 2]).mean()*100}%")
    print(f"Train True ATE: {(train_data[:, 3] - train_data[:, 2]).mean()*100:.4f}%")
    print(f"Val True ATE: {(val_data[:, 3] - val_data[:, 2]).mean()*100:.4f}%")
    print(f"Test True ATE: {(test_data[:, 3] - test_data[:, 2]).mean()*100:.4f}%")
    print()
    print(f"True yf ratio: {data[:, 4].mean()*100}%")
    print(f"Train True yf ratio: {train_data[:, 4].mean()*100:.4f}%")
    print(f"Val True yf ratio: {val_data[:, 4].mean()*100:.4f}%")
    print(f"Test True yf ratio: {test_data[:, 4].mean()*100:.4f}%")
    print()

    return train_data, val_data, test_data

def _construct_loader(dataset_name, data, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    assert dataset_name in _DATASET_CATALOG.keys(), "Dataset '{}' not supported".format(
        dataset_name
    )
    # Construct the dataset
    dataset = _DATASET_CATALOG[dataset_name](data)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(data):
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        data=data,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )


def construct_val_loader(data):
    """Val loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        data=data,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )


def construct_test_loader(data):
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        data=data,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
