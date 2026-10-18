"""
models.py -- configurable backbone(s) + head.

config.BACKBONES = ["efficientnet_b0"]                       -> single-backbone baseline
config.BACKBONES = ["efficientnet_b0","densenet121","resnet50"] -> fusion (ablation)

Fixes vs the previous version:
  * frozen backbones keep BatchNorm running stats fixed (config.FREEZE_BACKBONE_BN);
    model.train() used to silently update them with batches of 16.
  * DenseNet wrapper no longer uses an in-place ReLU (breaks Grad-CAM hooks).
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import config


class _DenseNetFeatures(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.features = net.features

    def forward(self, x):
        x = F.relu(self.features(x), inplace=False)
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)


def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name == "efficientnet_b0":
        net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        d = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, d
    if name == "densenet121":
        net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        return _DenseNetFeatures(net), net.classifier.in_features
    if name == "resnet50":
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        d = net.fc.in_features
        net.fc = nn.Identity()
        return net, d
    raise ValueError(f"Unknown backbone: {name}")


class MultiBackboneFundusClassifier(nn.Module):
    def __init__(self, backbone_names: List[str] = None, num_classes: int = config.NUM_CLASSES,
                 projection_dim: int = config.FUSION_PROJECTION_DIM, dropout: float = config.DROPOUT,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_names = list(backbone_names or config.BACKBONES)
        self.BACKBONE_NAMES = self.backbone_names          # used by explainability.py
        self.backbones, self.projections = nn.ModuleDict(), nn.ModuleDict()
        for n in self.backbone_names:
            bb, d = build_backbone(n, pretrained)
            self.backbones[n] = bb
            self.projections[n] = nn.Sequential(nn.Linear(d, projection_dim), nn.GELU())
        fused = projection_dim * len(self.backbone_names)
        self.head = nn.Sequential(nn.BatchNorm1d(fused), nn.Linear(fused, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, num_classes))
        self.set_backbones_trainable(False)

    # ---- BN handling ---------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if mode and config.FREEZE_BACKBONE_BN:
            for bb in self.backbones.values():
                for m in bb.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
        return self

    # ---- freezing ------------------------------------------------------------
    def set_backbones_trainable(self, trainable: bool) -> None:
        for bb in self.backbones.values():
            for p in bb.parameters():
                p.requires_grad = trainable

    def unfreeze_last_n_tensors(self, n) -> None:
        for bb in self.backbones.values():
            params = list(bb.parameters())
            keep = params if n is None else params[-n:]
            for p in params:
                p.requires_grad = False
            for p in keep:
                p.requires_grad = True

    # ---- forward ---------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.projections[n](self.backbones[n](x)) for n in self.backbone_names], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def get_target_layer(self, name: str) -> nn.Module:
        bb = self.backbones[name]
        if name == "efficientnet_b0":
            return bb.features[-1]
        if name == "densenet121":
            return bb.features               # output = norm5 feature map
        if name == "resnet50":
            return bb.layer4[-1]
        raise ValueError(name)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
