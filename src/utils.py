import os
import random
from time import gmtime, strftime
import yaml
import pprint

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def kld_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class Dict2Object:
    """
    Object that basically converts a dictionary of args
    to object of args. Purpose is to simplify calling the args
    (from args["lr"] to args.lr)
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_config(config_path: str, curr_time: str = None) -> Dict2Object:
    if curr_time is None:
        curr_time = strftime("%y_%m_%d_%H-%M-%S", gmtime())

    with open(config_path, 'r') as stream:
        cfg = yaml.load(stream)
    print("loaded config")
    print("="*90)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    print("="*90)

    args = Dict2Object(**cfg)
    args.output_dir += curr_time
    args.model_output_dir = args.output_dir + '/saved_models/'
    args.output_img_dir = args.output_dir + '/reconstructed_image_ep_'

    os.mkdir(args.output_dir)
    os.mkdir(args.model_output_dir)
    os.mkdir(args.output_img_dir)

    return args


def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(
        X=np.array([1]),
        Y=np.array([np.nan]),
        opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))


class UnNormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
