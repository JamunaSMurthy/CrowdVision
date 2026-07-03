"""Kernel-Graph Attention (Section 3.4, Eq. 6-7 of the paper).

Each encoded view E_s is mean-pooled into a descriptor c_s, which feeds a
learnable directed affinity graph A over the S view-nodes (a single-hop
GAT-style edge score, Eq. 6). KGA then aggregates the *fixed-size* kernel
summaries phi(K_j)^T V_j of every source view j into each target view i,
weighted by A_ij (Eq. 7). Because the kernel summary has shape (r, head_dim)
regardless of the source view's token count, views of different temporal
resolution combine directly with no resampling step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdvit.config import ModelConfig
from crowdvit.models.performer_attention import build_orthogonal_projection, favor_feature_map


class AffinityGraph(nn.Module):
    """Directed S x S adjacency over view descriptors. A[:, i, j] is the
    weight of source view j's contribution to target view i.
    """

    def __init__(self, dim: int, num_views: int, affinity_type: str):
        super().__init__()
        self.num_views = num_views
        self.affinity_type = affinity_type

        if affinity_type == "learnable":
            self.w_target = nn.Linear(dim, dim)
            self.w_source = nn.Linear(dim, dim)
            self.w_pair = nn.Linear(dim, dim)
            self.w_a = nn.Linear(dim, 1)
            self.act = nn.LeakyReLU(0.2)
        elif affinity_type == "fixed":
            self_weight = 0.6
            off = (1.0 - self_weight) / max(num_views - 1, 1)
            fixed = torch.full((num_views, num_views), off)
            fixed.fill_diagonal_(self_weight)
            self.register_buffer("fixed_graph", fixed)
        elif affinity_type not in ("uniform", "random", "none"):
            raise ValueError(f"Unknown affinity_type '{affinity_type}'")

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """descriptors: (B, S, D) -> A: (B, S, S)."""
        b, s, _ = descriptors.shape

        if self.affinity_type == "learnable":
            target_term = self.w_target(descriptors).unsqueeze(2)  # (B, S, 1, D)
            source_term = self.w_source(descriptors).unsqueeze(1)  # (B, 1, S, D)
            pair = descriptors.unsqueeze(2) * descriptors.unsqueeze(1)  # (B, S, S, D)
            pair_term = self.w_pair(pair)
            energy = self.w_a(self.act(target_term + source_term + pair_term)).squeeze(-1)
            return torch.softmax(energy, dim=-1)

        if self.affinity_type == "uniform":
            return descriptors.new_full((b, s, s), 1.0 / s)

        if self.affinity_type == "none":
            return torch.eye(s, device=descriptors.device, dtype=descriptors.dtype).expand(b, s, s)

        if self.affinity_type == "fixed":
            return self.fixed_graph.to(descriptors.dtype).unsqueeze(0).expand(b, -1, -1)

        if self.affinity_type == "random":
            concentration = torch.ones(s, device=descriptors.device)
            dist = torch.distributions.Dirichlet(concentration)
            return dist.sample((b, s)).to(descriptors.dtype)

        raise ValueError(f"Unknown affinity_type '{self.affinity_type}'")


class KernelGraphAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_views = cfg.num_views
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.hidden_dim = cfg.hidden_dim
        self.kernel_dim = cfg.kernel_dim

        self.qkv = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 3) for _ in range(self.num_views)]
        )
        self.out_proj = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(self.num_views)]
        )
        self.register_buffer(
            "projection_matrix",
            build_orthogonal_projection(cfg.kernel_dim, self.head_dim),
            persistent=False,
        )
        self.affinity = AffinityGraph(cfg.hidden_dim, self.num_views, cfg.affinity_type)

    @torch.no_grad()
    def redraw_projection_matrix(self) -> None:
        self.projection_matrix = build_orthogonal_projection(
            self.kernel_dim, self.head_dim, device=self.projection_matrix.device
        )

    def _project_view(self, x: torch.Tensor, view_idx: int):
        b, n, d = x.shape
        qkv = (
            self.qkv[view_idx](x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        phi_q = favor_feature_map(q, self.projection_matrix, is_query=True)
        phi_k = favor_feature_map(k, self.projection_matrix, is_query=False)
        kv_summary = torch.einsum("bhnr,bhnd->bhrd", phi_k, v)
        k_summary = phi_k.sum(dim=2)
        return phi_q, kv_summary, k_summary

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        batch_size = views[0].shape[0]
        phi_qs, kv_summaries, k_summaries, descriptors = [], [], [], []

        for s, view_tokens in enumerate(views):
            phi_q, kv_summary, k_summary = self._project_view(view_tokens, s)
            phi_qs.append(phi_q)
            kv_summaries.append(kv_summary)
            k_summaries.append(k_summary)
            descriptors.append(view_tokens.mean(dim=1))

        descriptors = torch.stack(descriptors, dim=1)  # (B, S, D)
        affinity = self.affinity(descriptors)  # (B, S, S)

        outputs = []
        for i in range(self.num_views):
            numerator = 0.0
            denominator = 0.0
            for j in range(self.num_views):
                weight = affinity[:, i, j].view(batch_size, 1, 1, 1)
                numerator = numerator + weight * torch.einsum(
                    "bhnr,bhrd->bhnd", phi_qs[i], kv_summaries[j]
                )
                denominator = denominator + weight.squeeze(-1) * torch.einsum(
                    "bhnr,bhr->bhn", phi_qs[i], k_summaries[j]
                )
            fused = numerator / (denominator.unsqueeze(-1) + 1e-6)
            fused = fused.permute(0, 2, 1, 3).reshape(
                batch_size, views[i].shape[1], self.hidden_dim
            )
            outputs.append(self.out_proj[i](fused))

        return outputs
