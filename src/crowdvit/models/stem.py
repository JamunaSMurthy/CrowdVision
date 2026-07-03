"""Lightweight 3D-convolution stem that produces the pooled clip descriptor
x-bar = GAP(Conv3D_stem(X)) used by the basis predictor and scale gate
(Eq. 2-3 of the paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdvit.config import ModelConfig


class Conv3DStem(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        mid_channels = cfg.stem_channels // 2
        self.net = nn.Sequential(
            nn.Conv3d(
                cfg.in_channels,
                mid_channels,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                mid_channels,
                cfg.stem_channels,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(cfg.stem_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(cfg.stem_channels, cfg.stem_descriptor_dim)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (B, T, C, H, W) -> x_bar: (B, stem_descriptor_dim)"""
        x = clip.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.net(x)
        x = self.pool(x).flatten(1)  # (B, stem_channels)
        return self.proj(x)
