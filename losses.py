"""losses.py -- imbalance handling. ONE strategy is active (config.IMBALANCE_STRATEGY)."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(train_labels, num_classes: int) -> torch.Tensor:
    counts = np.clip(np.bincount(train_labels, minlength=num_classes).astype(np.float32), 1, None)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def dataset_class_sample_weights(train_df: pd.DataFrame, max_ratio: float = 20.0) -> np.ndarray:
    """Within each dataset the classes get equal total mass; datasets keep their
    natural share of the epoch. Removes the per-dataset positive-rate shortcut.
    Weights are clipped to `max_ratio` x the smallest to avoid one tiny cell
    being repeated absurdly often."""
    cell = train_df.groupby(["dataset_name", "unified_class"])["image_path"].transform("count").to_numpy(float)
    ds_n = train_df.groupby("dataset_name")["image_path"].transform("count").to_numpy(float)
    w = (ds_n / len(train_df)) / cell
    w = np.clip(w, w.min(), w.min() * max_ratio)
    return (w / w.sum()).astype(np.float64)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        lp = logp.gather(1, targets[:, None]).squeeze(1)
        loss = -((1 - lp.exp()) ** self.gamma) * lp
        if self.alpha is not None:
            loss = self.alpha.to(logits.device).gather(0, targets) * loss
        return loss.mean()


def build_loss_fn(strategy: str, train_labels, num_classes: int, gamma: float = 2.0,
                  label_smoothing: float = 0.0):
    if strategy == "class_weighted_ce":
        return nn.CrossEntropyLoss(weight=compute_class_weights(train_labels, num_classes),
                                   label_smoothing=label_smoothing)
    if strategy in ("dataset_class_sampler", "weighted_sampler"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)      # balancing happens in the sampler
    if strategy == "focal_loss":
        return FocalLoss(alpha=compute_class_weights(train_labels, num_classes), gamma=gamma)
    raise ValueError(f"Unknown imbalance strategy: {strategy}")
