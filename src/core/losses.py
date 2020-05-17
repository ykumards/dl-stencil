"""Loss functions."""

import torch
import torch.nn as nn
from core.config import cfg


# Supported loss functions
_loss_funs = {
    "bce_logits": nn.BCEWithLogitsLoss,
    "mse_loss": nn.MSELoss,
    "ce_loss": nn.CrossEntropyLoss,
}


def kld_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def get_loss_fun_vae():
    """Retrieves the loss function."""
    assert (
        cfg.MODEL.LOSS_FUN in _loss_funs.keys()
    ), "Loss function '{}' not supported".format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN](), kld_loss


def get_loss_fun():
    """Retrieves the loss function."""
    assert (
        cfg.MODEL.LOSS_FUN in _loss_funs.keys()
    ), "Loss function '{}' not supported".format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]().to(cfg.DEVICE)


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
