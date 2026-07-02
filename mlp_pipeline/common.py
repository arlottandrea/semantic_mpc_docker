from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_detection_score(scores, midpoint, steepness):
    """Convert YOLO confidences to the centered score used by NMPC."""
    if not scores:
        return 0.0
    activation = 0.5 + 0.5 * np.tanh(steepness * (len(scores) - midpoint))
    return float(np.ceil(100 * (np.mean(scores) - 0.5) * activation) / 100)
