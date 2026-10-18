"""utils.py -- seeding, device, json, checkpoints, early stopping."""
import copy
import json
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_environment_info(device) -> None:
    print("=" * 70)
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print("=" * 70)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_checkpoint(model, path: str, epoch: int, metric: float, extra: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "best_metric": metric, **(extra or {})}, path)


def load_checkpoint_file(path: str, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter, self.should_stop = None, 0, False

    def step(self, value: float) -> bool:
        """True if `value` is a new best."""
        if self.best is None or value > self.best + self.min_delta:
            self.best, self.counter = value, 0
            return True
        self.counter += 1
        self.should_stop = self.counter >= self.patience
        return False


class ModelEma:
    """EMA of parameters + buffers, updated once per optimizer step (config.USE_EMA)."""

    def __init__(self, model, decay: float = 0.999):
        self.decay, self.step = decay, 0
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.step += 1
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])
