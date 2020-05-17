"""
Handles Tensorboard Logging.

Largely based on Pytorch Ignite's tb_logger:
https://github.com/pytorch/ignite/blob/master/ignite/contrib/handlers/tensorboard_logger.py
"""
import torch

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, *args, **kwargs):
        self.writer = SummaryWriter(*args, **kwargs)

    def write_scalar(self, tb_stats, cur_epoch, keys_to_use=["loss"], tag=""):
        for key, value in tb_stats.items():
            if key in keys_to_use:
                self.writer.add_scalar("{}/{}".format(tag, key), value, cur_epoch)

    def write_model_weight_hist(self, model, cur_epoch, every=10, tag=""):
        if cur_epoch % every == 0:
            tag_prefix = "{}/".format(tag) if tag else ""
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                self.writer.add_histogram(
                    tag="{}weights/{}".format(tag_prefix, name),
                    values=p.data.detach().cpu().numpy(),
                    global_step=cur_epoch,
                )

    def write_model_grad_hist(self, model, cur_epoch, every=10, tag=""):
        if cur_epoch % every == 0:
            tag_prefix = "{}/".format(tag) if tag else ""
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                self.writer.add_histogram(
                    tag="{}grads/{}".format(tag_prefix, name),
                    values=p.grad.detach().cpu().numpy(),
                    global_step=cur_epoch,
                )

    def write_weight_scalar(self, model, cur_epoch, every=1, tag=""):
        # TODO
        pass

    def write_optim_scalar(self, optimizer, cur_epoch, param_name="lr", tag=None):
        tag_prefix = "{}/".format(tag) if tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, param_name, i): float(
                param_group[param_name]
            )
            for i, param_group in enumerate(optimizer.param_groups)
        }
        for k, v in params.items():
            self.writer.add_scalar(k, v, cur_epoch)

    def close(self):
        self.writer.close()
