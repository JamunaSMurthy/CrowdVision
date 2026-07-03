"""L_view: cross-view consistency regularization.

The paper describes L_view qualitatively as encouraging KGA-aligned temporal
views to "remain semantically compatible without collapsing into identical
representations," again without a closed form. We realize this as two terms:

  1. an alignment term that pulls every pair of post-fusion view descriptors
     together via cosine similarity, and
  2. a collapse-avoidance term (in the spirit of VICReg's variance term) that
     penalizes each view descriptor's per-dimension batch standard deviation
     falling below a target floor, so the alignment term alone cannot drive
     all views to an identical (collapsed) representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewConsistencyLoss(nn.Module):
    def __init__(self, collapse_weight: float = 1.0, target_std: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.collapse_weight = collapse_weight
        self.target_std = target_std
        self.eps = eps

    def forward(self, view_descriptors: list[torch.Tensor]) -> torch.Tensor:
        num_views = len(view_descriptors)
        if num_views < 2:
            return view_descriptors[0].new_zeros(())

        alignment = view_descriptors[0].new_zeros(())
        num_pairs = 0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                a = F.normalize(view_descriptors[i], dim=-1)
                b = F.normalize(view_descriptors[j], dim=-1)
                alignment = alignment + (1.0 - (a * b).sum(dim=-1)).mean()
                num_pairs += 1
        alignment = alignment / num_pairs

        collapse_penalty = view_descriptors[0].new_zeros(())
        for descriptor in view_descriptors:
            std = torch.sqrt(descriptor.var(dim=0) + self.eps)
            collapse_penalty = collapse_penalty + F.relu(self.target_std - std).mean()
        collapse_penalty = collapse_penalty / num_views

        return alignment + self.collapse_weight * collapse_penalty
