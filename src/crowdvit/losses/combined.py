"""Combined training objective: wires the classification, temporal-smoothness,
and cross-view-consistency losses together into a single weighted sum
``total = ce + lambda_temp * temp + lambda_view * view``, with the temporal
and view terms individually toggleable via ``LossConfig`` (used by the
loss-ablation configs in ``configs/ablation_no_*.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from crowdvit.config import LossConfig
from crowdvit.losses.classification_loss import ClassificationLoss
from crowdvit.losses.temporal_smoothness import TemporalSmoothnessLoss
from crowdvit.losses.view_consistency import ViewConsistencyLoss
from crowdvit.models.crowdvit import CrowdViTOutput


@dataclass
class CrowdViTLossOutput:
    """The total loss plus each individual term, so a training loop can log
    them separately without recomputing anything."""

    total: torch.Tensor
    ce: torch.Tensor
    temp: torch.Tensor
    view: torch.Tensor


class CrowdViTLoss(nn.Module):
    """Computes ``CrowdViTLossOutput`` from a model forward pass and its
    targets, respecting the ``use_temporal_loss`` / ``use_view_loss`` flags
    in ``cfg`` (a disabled term is reported as zero rather than skipped, so
    downstream logging code doesn't need to branch on which terms are on)."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.classification_loss = ClassificationLoss(cfg.label_smoothing)
        self.temporal_loss = TemporalSmoothnessLoss()
        self.view_loss = ViewConsistencyLoss(cfg.view_collapse_weight, cfg.view_collapse_target_std)

    def forward(self, output: CrowdViTOutput, targets: torch.Tensor) -> CrowdViTLossOutput:
        ce = self.classification_loss(output.logits, targets)

        temp = output.logits.new_zeros(())
        if self.cfg.use_temporal_loss:
            temp = self.temporal_loss(output.frame_sequence)

        view = output.logits.new_zeros(())
        if self.cfg.use_view_loss:
            view = self.view_loss(output.view_descriptors)

        total = ce + self.cfg.lambda_temp * temp + self.cfg.lambda_view * view
        return CrowdViTLossOutput(total=total, ce=ce, temp=temp, view=view)
