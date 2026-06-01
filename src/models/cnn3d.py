"""
cnn3d.py
--------
CNN 3D compacta para clasificacion binaria de MRI cerebral T1/T2.
"""

from __future__ import annotations

import torch
from torch import nn


def _norm(num_channels: int) -> nn.Module:
    """GroupNorm con 8 grupos (o menos si el canal es pequeño).

    Se usa en vez de BatchNorm3d porque con batch_size=1 las estadisticas
    por batch son ruidosas y los running stats no convergen, generando un
    gap masivo entre train y val.
    """
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BrainTumorCNN3D(nn.Module):
    """Modelo 3D pequeno para clasificacion binaria: tumor frente a no tumor."""

    def __init__(
        self,
        in_channels: int = 2,
        n_classes: int = 1,
        base_channels: int = 12,
        dropout: float = 0.25,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.features = nn.Sequential(
            ConvBlock3D(in_channels, c1, dropout=0.0),
            nn.MaxPool3d(2),
            ConvBlock3D(c1, c2, dropout=dropout * 0.5),
            nn.MaxPool3d(2),
            ConvBlock3D(c2, c3, dropout=dropout),
            nn.MaxPool3d(2),
            ConvBlock3D(c3, c4, dropout=dropout),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, c4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def build_cnn3d(**kwargs) -> BrainTumorCNN3D:
    return BrainTumorCNN3D(**kwargs)
