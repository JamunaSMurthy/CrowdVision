"""Two-layer MLP prediction head (Eq. "MLP(h)" in Section 3.5)."""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdvit.config import ModelConfig


class MLPHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.hidden_dim * 2
        self.fc1 = nn.Linear(cfg.hidden_dim, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.head_dropout)
        self.fc2 = nn.Linear(hidden, cfg.num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(h))))
