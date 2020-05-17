"""Dataset paths."""

import os

from core.config import cfg

_DEF_DATA_DIR = "/data"

# Data paths
_paths = {"news": _DEF_DATA_DIR + "/mnist/"}


def has_data_path(dataset_name):
    """Determines if the dataset has a data path."""
    return dataset_name in _paths.keys()


def get_data_path(dataset_name):
    """Retrieves data path for the dataset."""
    return cfg.PATHS.DATAPATH


def register_path(name, path):
    """Registers a dataset path dynamically."""
    _paths[name] = path
