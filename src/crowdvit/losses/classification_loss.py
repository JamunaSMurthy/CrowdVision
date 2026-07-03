"""Clip-level action classification loss: plain cross-entropy between the
model's predicted class logits and the ground-truth action label, with
optional label smoothing (this is the L_ce term of the combined objective
in ``losses/combined.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
