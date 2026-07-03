"""L_temp: temporal smoothness regularization.

The paper describes L_temp qualitatively as discouraging "unstable
frame-to-frame representation changes" without a closed form. We realize it
as the mean squared difference between consecutive per-window descriptors of
the finest tubelet view (``CrowdViTOutput.frame_sequence``), which is the
representation closest to raw per-frame content available in the model. This
is an ordinary differentiable regularizer with no inference-time cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSmoothnessLoss(nn.Module):
    def forward(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        """frame_sequence: (B, num_windows, D)."""
        if frame_sequence.shape[1] < 2:
            return frame_sequence.new_zeros(())
        deltas = frame_sequence[:, 1:] - frame_sequence[:, :-1]
        return deltas.pow(2).mean()
