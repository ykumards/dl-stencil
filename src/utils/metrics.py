import numpy as np
import torch
import torch.nn as nn
import ignite.metrics as ignite_metrics

from core.config import cfg
import core.losses as losses

# Number of bytes in a megabyte
_B_IN_MB = 1024 * 1024


def num_correct_fun(preds, labels):
    """Computes the number of correct predictions."""
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find number of correct predictions
    num_correct = (preds == labels).float().sum()
    return num_correct


def label_errors(preds, labels):
    """Computes the label error."""
    num_correct = num_correct_fun(preds, labels)
    return (1.0 - num_correct / preds.size(0)) * 100.0


def pehe(ypred1, ypred0, y1, y0):
    return torch.sqrt(torch.mean(torch.pow((y1 - y0) - (ypred1 - ypred0), 2)))


def abs_ate(ypred1, ypred0, y1, y0):
    true_ite = y1 - y0
    return torch.abs(torch.mean(ypred1 - ypred0) - torch.mean(true_ite))


# TODO weirdly defined
# def rmse_ite(ypred1, ypred0, y1, y0, t):
#     true_ite = y1 - y0
#     pred_ite = torch.zeros_like(true_ite)
#     idx1, idx0 = torch.where(t == 1), torch.where(t == 0)
#     ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
#     pred_ite[idx1] = ite1
#     pred_ite[idx0] = ite0
#     return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))


def label_accuracies(preds, labels):
    """Computes the label accuracy."""
    num_correct = num_correct_fun(preds, labels)
    return (num_correct / preds.size(0)) * 100.0


def params_count(model):
    """Computes the number of parameters."""
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_MB


def acts_count(model):
    """Computes the number of activations statically."""
    count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            count += m.out_features
    return count


def flops_count(model):
    """Computes the number of flops statically."""
    count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            count += m.in_features * m.out_features + m.bias.numel()
    return count
