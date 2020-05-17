"""Dataset paths."""

import os

_DEF_DATA_DIR = "/data"

# Data paths
_paths = {"news": _DEF_DATA_DIR + "/mnist/"}


def has_data_path(dataset_name):
    """Determines if the dataset has a data path."""
    return dataset_name in _paths.keys()


def get_data_path(dataset_name):
    """Retrieves data path for the dataset."""
    return _paths[dataset_name]


def register_path(name, path):
    """Registers a dataset path dynamically."""
    _paths[name] = path
