import torch

from core.config import cfg
import utils.logging as lu

from models.resnet50 import Resnet50


logger = lu.get_logger(__name__)

# Supported models
_models = {
    "resnet50": Resnet50,
}


def build_model():
    """Builds the model."""
    assert cfg.MODEL.TYPE in _models.keys(), "Model type '{}' not supported".format(
        cfg.MODEL.TYPE
    )
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"
    # Construct the model
    model = _models[cfg.MODEL.TYPE]()
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    # TODO: add support for multi-gpu
    return model


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor