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


class CFRMeter(object):
    """Measures counterfactual metrics."""
    def __init__(self, datalen):
        super().__init__()
        # WARNING: Stores everything in working memory
        self.datalen = datalen
        self.y0pred_arr = np.ones((datalen,))*-1000
        self.y1pred_arr = np.ones((datalen,))*-1000
        self.y0_arr = np.ones((datalen,))*-1000
        self.y1_arr = np.ones((datalen,))*-1000
        self.offset = 0

    def reset(self):
        self.y0pred_arr = np.ones((self.datalen,))*-1000
        self.y1pred_arr = np.ones((self.datalen,))*-1000
        self.y0_arr = np.ones((self.datalen,))*-1000
        self.y1_arr = np.ones((self.datalen,))*-1000
        self.offset = 0

    def add_values(self, y1_pred, y0_pred, y1, y0):
        batch_size = y1_pred.shape[0]
        self.y1pred_arr[self.offset:self.offset+batch_size] = y1_pred
        self.y0pred_arr[self.offset:self.offset+batch_size] = y0_pred
        self.y1_arr[self.offset:self.offset+batch_size] = y1
        self.y0_arr[self.offset:self.offset+batch_size] = y0
        self.offset += batch_size

    def get_ite_true(self):
        return self.y1_arr - self.y0_arr

    def get_ite_pred(self):
        return self.y1pred_arr - self.y0pred_arr

    def get_pehe(self):
        true_ite = self.get_ite_true()
        pred_ite = self.get_ite_pred()
        return np.sqrt(np.mean(np.square((true_ite) - (pred_ite))))

    def get_abs_ate(self):
        true_ite = self.get_ite_true()
        pred_ite = self.get_ite_pred()
        return np.abs(np.mean(pred_ite) - np.mean(true_ite))


class Meter(object):
    "Measures and handles training/eval stats"

    def __init__(self, epoch_iters, batch_size, mode='train'):
        assert mode in ['train', 'valid', 'test'],\
            "mode can only be train, val or test"

        self.mode = mode
        self.epoch_iters = epoch_iters
        self.max_iter = epoch_iters
        if self.mode == 'train':
            self.max_iter *= cfg.OPTIM.MAX_EPOCH

        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window)
        self.mb_label_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples
        self.num_mis = 0
        self.num_samples = 0

        self.cfr_meter = CFRMeter(epoch_iters * batch_size)

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_label_err.reset()
        self.num_mis = 0
        self.num_samples = 0
        self.cfr_meter.reset()

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, label_err, loss, mb_size, lr=None):
        # Current minibatch stats
        self.mb_label_err.add_value(label_err)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_mis += label_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            "_type": self.mode + "_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "label_err": self.mb_label_err.get_win_median(),
            "loss": self.loss.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        if self.mode == 'train':
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
        label_err = self.num_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        avg_pehe = self.cfr_meter.get_pehe()
        avg_abs_ate = self.cfr_meter.get_abs_ate()
        ate_pred = self.cfr_meter.get_ite_pred().mean()
        ate_true = self.cfr_meter.get_ite_true().mean()
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "label_err": label_err,
            "loss": avg_loss,
            "pehe": avg_pehe,
            "abs_ate": avg_abs_ate,
            "ate_pred": ate_pred*100,
            "ate_true": ate_true*100,
            "mem": int(np.ceil(mem_usage)),
        }
        if self.mode == 'train':
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
        self,
        cur_epoch,
        keys_to_print=['label_err', 'loss', 'pehe', 'abs_ate', 'ate_pred', 'ate_true']
    ):
        print_str = self.mode.upper() + ": "
        stats = self.get_epoch_stats(cur_epoch)
        print_str += f"epoch : {stats['epoch']} "

        for key, value in stats.items():
            if key in keys_to_print:
                print_str += f"{key}: {value:.4f} "
        tqdm.write(print_str)