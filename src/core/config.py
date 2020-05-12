"""Configuration file (powered by YACS)."""

import os
from datetime import datetime
import torch

from yacs.config import CfgNode as CN


# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type
_C.MODEL.TYPE = ""

# Whether to use Tarnet ad DragonNet (ie. predict p(t|x) as well)
_C.MODEL.IS_DRAGON = False
# Model input dimension
_C.MODEL.INPUT_DIM = 1
# Dimension of continuous features
_C.MODEL.DIM_CONT = 0
# Dimension of binary+categorical features
_C.MODEL.DIM_BIN = 0

# Size of latent dimension
_C.MODEL.Z_DIM = 64

# Following are used in CEVAE
# Decoder X network, hidden dimension
_C.MODEL.DEC_X_HIDDEN_DIM = 10
# Decoder X network, number of hidden layers
_C.MODEL.DEC_X_NUM_HIDDEN = 1
# Decoder Y network, hidden dimension
_C.MODEL.DEC_Y_HIDDEN_DIM = 10
# Decoder Y network, number of hidden layers
_C.MODEL.DEC_Y_NUM_HIDDEN = 1

# Encoder Y network, hidden dimension
_C.MODEL.ENC_Y_HIDDEN_DIM = 10
# Encoder Y network, number of hidden_layers
_C.MODEL.ENC_Y_NUM_HIDDEN = 1
# Encoder Z network, hidden dimension
_C.MODEL.ENC_Z_HIDDEN_DIM = 10
# Encoder Z network, number of hidden_layers
_C.MODEL.ENC_Z_NUM_HIDDEN = 1

# number of experts on decisions (d) for MoE VAE
_C.MODEL.NUM_EXPERTS = 1

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = "bce_logits"


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 2

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Grad clipping value for exploding gradients
_C.OPTIM.GRAD_CLIP_T = 1e4

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""

# Patience for early stopping
_C.TRAIN.ES_PATIENCE = 0

# Loss threshold for early stopping
_C.TRAIN.ES_THRESHOLD = 0.

# Switch off KL term in VAE
_C.TRAIN.SWITCHOFF_KL = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200

# Weights to use for testing
_C.TEST.WEIGHTS = ""

# Which type of ITE to compute
_C.TEST.WHICH_ITE = -1

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 4

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Memory options
# ---------------------------------------------------------------------------- #
_C.MEM = CN()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True


# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = True


# ---------------------------------------------------------------------------- #
# Precise timing options
# ---------------------------------------------------------------------------- #
_C.PREC_TIME = CN()

# Perform precise timing at the start of training
_C.PREC_TIME.ENABLED = False

# Total mini-batch size
_C.PREC_TIME.BATCH_SIZE = 128

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ---------------------------------------------------------------------------- #
# Ignite Handler options
# ---------------------------------------------------------------------------- #
_C.IGNITE_HANDLERS = CN()

# Model Checkpoint
# Number of checkpoints to save
_C.IGNITE_HANDLERS.CKPT_NSAVED = 3
# Checkpoint save interval
_C.IGNITE_HANDLERS.CKPT_SAVE_INTERVAL = 5  # epochs
# Checkpoint file prefix
_C.IGNITE_HANDLERS.CKPT_PREFIX = ""


# ---------------------------------------------------------------------------- #
# Path options
# ---------------------------------------------------------------------------- #
_C.PATHS = CN()
# Output directory parent folder
_C.PATHS.OUT_DIR = ""
# Experiment name
_C.PATHS.EXPERIMENT_NAME = ""
# Get current timestamp
_C.PATHS.TIMESTAMP = "at_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
# Outdirectory for TB logging
_C.PATHS.TB_OUT_DIR = os.path.join(_C.PATHS.OUT_DIR, "tb_logs", _C.PATHS.TIMESTAMP)
# Outdirectory for model checkpoints
_C.PATHS.MODEL_OUT_DIR = os.path.join(_C.PATHS.OUT_DIR, "saved_models", _C.PATHS.TIMESTAMP)

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# tune lr if set to true, tunes model between a range of lr values
_C.TUNE_LR = False

# Choose device type
_C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 100

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Tensorboard logging flag
_C.IS_TB_LOG = True

# Frequency for logging gradient histograms in TB
_C.TB_LOG_GRAD_INTV = 500  # iteration


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    assert (
        not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0
    ), "The first lr step must start at 0"
    assert _C.TRAIN.SPLIT in [
        "train",
        "val",
        "test",
    ], "Train split '{}' not supported".format(_C.TRAIN.SPLIT)
    assert (
        _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0
    ), "Train mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TEST.SPLIT in [
        "train",
        "val",
        "test",
    ], "Test split '{}' not supported".format(_C.TEST.SPLIT)
    assert (
        _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0
    ), "Test mini-batch size should be a multiple of NUM_GPUS."
    assert (
        not _C.BN.USE_PRECISE_STATS or _C.NUM_GPUS == 1
    ), "Precise BN stats computation not verified for > 1 GPU"
    assert _C.LOG_DEST in [
        "stdout",
        "file",
    ], "Log destination '{}' not supported".format(_C.LOG_DEST)
    assert (
        not _C.PREC_TIME.ENABLED or _C.NUM_GPUS == 1
    ), "Precise iter time computation not verified for > 1 GPU"


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.PATHS.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)
