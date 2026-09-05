"""PixelFlow model loading (vendored pixelflow package, fp32, eval)."""
import os
import torch
from omegaconf import OmegaConf
from pixelflow.utils import config as config_utils


def load_model(model_dir, device):
    """Returns (config, model). model_dir holds config.yaml + model.pt (PixelFlow class-conditional 256, 4 stages)."""
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return config, model
