"""Fusion-mechanism variants used in the "Fusion mechanism ablation" table.

``kga`` (Kernel-Graph Attention) is the proposed mechanism; the remaining
four variants are ablation baselines that isolate what KGA actually buys:

  - concat_only:        no cross-view interaction at all before the tokens
                         reach the Global Fusion Encoder.
  - performer_only:     cross-view mixing via one shared Performer attention
                         layer run over the concatenation of all views'
                         tokens, i.e. kernelized attention without the
                         learned affinity graph.
  - graph_only:         the learned affinity graph mixes pooled view
                         descriptors and broadcasts the result back onto
                         every token of the target view, without any
                         kernelized token-level attention.
  - full_cross_attention: dense quadratic softmax attention over the
                         concatenation of all views' tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdvit.config import ModelConfig
from crowdvit.models.kga import AffinityGraph, KernelGraphAttention
from crowdvit.models.performer_attention import PerformerSelfAttention, FullSelfAttention


class ConcatOnlyFusion(nn.Module):
    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        return views


class _WholeSequenceFusion(nn.Module):
    """Shared base for fusion variants that operate on the concatenation of
    all views' tokens as one sequence, then split back into per-view chunks.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        sizes = [v.shape[1] for v in views]
        merged = torch.cat(views, dim=1)
        fused = self.attn(merged)
        return list(torch.split(fused, sizes, dim=1))


class PerformerOnlyFusion(_WholeSequenceFusion):
    def __init__(self, cfg: ModelConfig):
        super().__init__(
            PerformerSelfAttention(
                cfg.hidden_dim, cfg.num_heads, cfg.kernel_dim, cfg.attn_dropout, cfg.proj_dropout
            )
        )


class FullCrossAttentionFusion(_WholeSequenceFusion):
    def __init__(self, cfg: ModelConfig):
        super().__init__(
            FullSelfAttention(cfg.hidden_dim, cfg.num_heads, cfg.attn_dropout, cfg.proj_dropout)
        )


class GraphOnlyFusion(nn.Module):
    """Uses the learned affinity graph to mix pooled descriptors only —
    no kernelized token-level attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_views = cfg.num_views
        self.affinity = AffinityGraph(cfg.hidden_dim, cfg.num_views, cfg.affinity_type)
        self.residual_proj = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.num_views)]
        )

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        descriptors = torch.stack([v.mean(dim=1) for v in views], dim=1)  # (B, S, D)
        affinity = self.affinity(descriptors)  # (B, S, S)
        aggregated = torch.einsum("bij,bjd->bid", affinity, descriptors)  # (B, S, D)
        outputs = []
        for i, view_tokens in enumerate(views):
            bias = self.residual_proj[i](aggregated[:, i, :]).unsqueeze(1)
            outputs.append(view_tokens + bias)
        return outputs


def build_fusion(cfg: ModelConfig) -> nn.Module:
    if cfg.fusion_type == "kga":
        return KernelGraphAttention(cfg)
    if cfg.fusion_type == "concat_only":
        return ConcatOnlyFusion()
    if cfg.fusion_type == "performer_only":
        return PerformerOnlyFusion(cfg)
    if cfg.fusion_type == "graph_only":
        return GraphOnlyFusion(cfg)
    if cfg.fusion_type == "full_cross_attention":
        return FullCrossAttentionFusion(cfg)
    raise ValueError(f"Unknown fusion_type '{cfg.fusion_type}'")
