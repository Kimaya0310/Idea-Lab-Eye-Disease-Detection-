"""
explainability.py -- Grad-CAM per backbone.

Uses a forward hook + a tensor hook on the layer output (NOT
register_full_backward_hook, which raises on layers followed by in-place ops
such as DenseNet's ReLU).
"""
import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model, self.activations, self.gradients = model, None, None
        self._h = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        self.gradients = None
        logits = self.model(x)
        logits[:, class_idx].sum().backward()
        if self.gradients is None:
            raise RuntimeError("No gradients captured: input tensor must require grad.")
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        cam = cam - cam.min()
        return cam / (cam.max() + 1e-8)

    def remove(self):
        self._h.remove()


def overlay(img_uint8: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.clip(img_uint8 * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def generate_multi_backbone_cam(model, x: torch.Tensor, display_img: np.ndarray, class_idx: int,
                                names: List[str] = None) -> dict:
    names = names or model.BACKBONE_NAMES
    maps = {}
    for n in names:
        cam = GradCAM(model, model.get_target_layer(n))
        try:
            maps[n] = cam(x.clone().requires_grad_(True), class_idx)
        finally:
            cam.remove()
    res = {n: overlay(display_img, m) for n, m in maps.items()}
    if len(maps) > 1:
        comb = np.mean(list(maps.values()), axis=0)
        res["combined"] = overlay(display_img, comb / (comb.max() + 1e-8))
    return res


def save_gradcam_figure(original, overlays: dict, title: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1 + len(overlays), figsize=(4 * (1 + len(overlays)), 4.4), squeeze=False)
    ax[0][0].imshow(original); ax[0][0].set_title("Preprocessed input"); ax[0][0].axis("off")
    for i, (n, o) in enumerate(overlays.items(), 1):
        ax[0][i].imshow(o); ax[0][i].set_title(n); ax[0][i].axis("off")
    fig.suptitle(title); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140); plt.close()
