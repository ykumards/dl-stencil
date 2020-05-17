"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
import torch
import torch.utils.data

import utils.logging as lu
import dataset.transforms as transforms
from core.config import cfg

logger = lu.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [125.3, 123.0, 113.9]
_SD = [63.0, 62.1, 66.7]


class Cifar10(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, datapath, split):
        assert os.path.exists(datapath), "Data path '{}' not found".format(datapath)
        assert split in ["train", "test", "valid"], "Split '{}' not supported for cifar".format(
            split
        )
        logger.info("Constructing CIFAR-10 {}...".format(split))
        self._datapath = datapath
        self._split = split
        # Data format:
        #   self._inputs - (split_size, 3, im_size, im_size) ndarray
        #   self._labels - split_size list
        self._inputs, self._labels = self._load_data()

    def _load_batch(self, batch_path):
        with open(batch_path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], d[b"labels"]

    def _load_data(self):
        """Loads data in memory."""
        logger.info("{} data path: {}".format(self._split, self._datapath))
        # Compute data batch names
        if self._split == "train":
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._datapath, batch_name)
            inputs_batch, labels_batch = self._load_batch(batch_path)
            inputs.append(inputs_batch)
            labels += labels_batch
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE))
        return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = transforms.color_norm(im, _MEAN, _SD)
        if self._split == "train":
            im = transforms.horizontal_flip(im=im, p=0.5)
            im = transforms.random_crop(im=im, size=cfg.TRAIN.IM_SIZE, pad_size=4)
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]
