"""Meters."""

import datetime
from collections import deque

import numpy as np
from tqdm import tqdm
import utils.logging as lu
import utils.metrics as metrics
from core.config import cfg
from utils.timer import Timer


def eta_str(eta_td):
    """Converts an eta timedelta to a fixed-width string format."""
    days = eta_td.days
    hrs, rem = divmod(eta_td.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class Meter(object):
    "Measures and handles training/eval stats"

    def __init__(self, epoch_iters, batch_size, scalar_metrics=["loss"], mode="train"):
        assert mode in ["train", "valid", "test"], "mode can only be train, val or test"

        self.mode = mode
        self.epoch_iters = epoch_iters
        self.max_iter = epoch_iters
        if self.mode == "train":
            self.max_iter *= cfg.OPTIM.MAX_EPOCH

        self.iter_timer = Timer()
        self.metrics_meters = {metric_name: [ScalarMeter(cfg.LOG_PERIOD), 0.] for metric_name in scalar_metrics}
        self.lr = None
        # Current minibatch errors (smoothed over a window)
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        # TODO refactor into meters
        for k, (sc_meter, _) in self.metrics_meters.items():
            sc_meter.reset()
            self.metrics_meters[k][1] = 0.

        self.lr = None
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, update_metrics_values, mb_size, lr=None):
        # Current minibatch stats
        for k, value in update_metrics_values.items():
            if k in self.metrics_meters:
                self.metrics_meters[k][0].add_value(value)
                self.metrics_meters[k][1] += value * mb_size

        self.lr = lr
        # Aggregate stats
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            "_type": self.mode + "_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "mem": int(np.ceil(mem_usage)),
        }
        for k, (sc_meter, _) in self.metrics_meters.items():
            stats[k] = sc_meter.get_win_avg()

        if self.mode == "train":
            eta_sec = self.iter_timer.average_time * (
                self.max_iter - (cur_epoch * self.epoch_iters + cur_iter + 1)
            )
            eta_td = datetime.timedelta(seconds=int(eta_sec))
            stats["eta"] = eta_str(eta_td)
            stats["lr"] = self.lr

        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        lu.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "mem": int(np.ceil(mem_usage)),
        }
        for k, (_, value) in self.metrics_meters.items():
            stats[k] = value / self.num_samples

        if self.mode == "train":
            eta_sec = self.iter_timer.average_time * (
                self.max_iter - (cur_epoch + 1) * self.epoch_iters
            )
            eta_td = datetime.timedelta(seconds=int(eta_sec))
            stats["lr"] = self.lr
            stats["eta"] = eta_str(eta_td)

        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        lu.log_json_stats(stats)

    def print_epoch_stats(
        self, cur_epoch, keys_to_print=None,
    ):
        if keys_to_print is None:
            keys_to_print = list(self.metrics_meters.keys())
        print_str = self.mode.upper() + ": "
        stats = self.get_epoch_stats(cur_epoch)
        print_str += f"epoch : {stats['epoch']} "

        for key, value in stats.items():
            if key in keys_to_print:
                print_str += f"{key}: {value:.4f} "
        tqdm.write(print_str)
