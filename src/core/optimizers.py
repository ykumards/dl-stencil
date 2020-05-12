"""Optimizer."""

import torch
from core.config import cfg
import utils.lr_policy as lr_policy


def construct_optimizer(model):
    """Constructs the optimizer."""
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    bn_weight_decay = (
        cfg.BN.CUSTOM_WEIGHT_DECAY
        if cfg.BN.USE_CUSTOM_WEIGHT_DECAY
        else cfg.OPTIM.WEIGHT_DECAY
    )
    optim_params = [
        {"params": bn_params, "weight_decay": bn_weight_decay},
        {"params": non_bn_parameters, "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )

    return torch.optim.SGD(
        optim_params,
        lr=cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        dampening=cfg.OPTIM.DAMPENING,
        nesterov=cfg.OPTIM.NESTEROV,
    )


def get_lr_scheduler(optimizer, gamma=0.975):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch (as specified by the lr policy)."""
    return lr_policy.get_epoch_lr(cur_epoch)


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr