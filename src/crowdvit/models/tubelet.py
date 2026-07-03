"""Adaptive Multi-Scale Tubelet Views (Section 3.2 of the paper).

Implements the two independent points of adaptivity described in the paper:

  (i)  the intra-window basis b_s(X) = softmax(W_s x_bar + beta_s), which
       decides *how* the k_s frames inside a temporal window are combined
       into a tubelet token (Eq. 2), and
  (ii) the scale gate alpha_s(X), which decides *how much* each resulting
       view contributes downstream (Eq. 3).

``view_mode`` selects three variants used throughout the paper's ablations:
  - "adaptive": b_s is predicted per-clip (the deployed model).
  - "fixed":    b_s is replaced by a uniform weight vector (tau=k_s uniform
                pooling), used only as an ablation baseline.
  - "pyramid":  views are not built independently; the finest view is built
                first and coarser views are obtained by progressively
                average-pooling adjacent pairs of the next-finer view's
                tokens, used only as an ablation baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdvit.config import ModelConfig


def patchify(clip: torch.Tensor, patch_size: int) -> torch.Tensor:
    """clip: (B, T, C, H, W) -> raw patches: (B, T, P, patch_dim)."""
    b, t, c, h, w = clip.shape
    gh, gw = h // patch_size, w // patch_size
    x = clip.view(b, t, c, gh, patch_size, gw, patch_size)
    x = x.permute(0, 1, 3, 5, 2, 4, 6)  # B,T,gh,gw,C,ph,pw
    x = x.reshape(b, t, gh * gw, c * patch_size * patch_size)
    return x


class ScaleGate(nn.Module):
    """Eq. 3: input-dependent gate deciding each view's overall contribution."""

    def __init__(self, descriptor_dim: int, num_views: int):
        super().__init__()
        self.head = nn.Linear(descriptor_dim, num_views)

    def forward(self, x_bar: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(x_bar), dim=-1)  # (B, S)


class _SingleViewBasis(nn.Module):
    """Basis predictor + projection for one temporal view, adaptive or fixed."""

    def __init__(
        self, patch_dim: int, hidden_dim: int, descriptor_dim: int, window_size: int, adaptive: bool
    ):
        super().__init__()
        self.window_size = window_size
        self.adaptive = adaptive
        self.proj = nn.Linear(patch_dim, hidden_dim)
        if adaptive:
            self.basis_head = nn.Linear(descriptor_dim, window_size)
        else:
            self.register_buffer("uniform_basis", torch.full((window_size,), 1.0 / window_size))

    def forward(self, raw_patches: torch.Tensor, x_bar: torch.Tensor) -> torch.Tensor:
        """raw_patches: (B, T, P, patch_dim) -> tokens: (B, N_s, hidden_dim)."""
        b, t, p, patch_dim = raw_patches.shape
        k = self.window_size
        num_windows = t // k
        windows = raw_patches.view(b, num_windows, k, p, patch_dim)

        if self.adaptive:
            basis = torch.softmax(self.basis_head(x_bar), dim=-1)  # (B, k)
        else:
            basis = self.uniform_basis.unsqueeze(0).expand(b, -1)

        weights = basis.view(b, 1, k, 1, 1)
        combined = (windows * weights).sum(dim=2)  # (B, num_windows, P, patch_dim)
        combined = combined.reshape(b, num_windows * p, patch_dim)
        return self.proj(combined)


class MultiScaleTubeletEmbed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.window_sizes = list(cfg.view_window_sizes)
        self.view_mode = cfg.view_mode
        patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size

        if self.view_mode == "pyramid":
            # Build the finest (smallest window) view from raw patches; every
            # coarser view is then derived by pooling pairs of the previous
            # view's tokens, so only the finest view needs a basis/proj.
            finest_k = min(self.window_sizes)
            self.finest_window = finest_k
            self.pyramid_proj = nn.Linear(patch_dim, cfg.hidden_dim)
            self.pyramid_ratios = [max(1, k // finest_k) for k in sorted(self.window_sizes)]
        else:
            adaptive = self.view_mode == "adaptive"
            self.view_embeds = nn.ModuleList(
                [
                    _SingleViewBasis(
                        patch_dim, cfg.hidden_dim, cfg.stem_descriptor_dim, k, adaptive
                    )
                    for k in self.window_sizes
                ]
            )

        self.scale_gate = ScaleGate(cfg.stem_descriptor_dim, len(self.window_sizes))

    def _pyramid_forward(self, raw_patches: torch.Tensor) -> list[torch.Tensor]:
        """Build the finest view from raw patches, then derive each coarser
        view by average-pooling adjacent pairs of the next-finer view's
        token grid (B, num_windows, P, d), halving num_windows per step.
        """
        b, t, p, patch_dim = raw_patches.shape
        k0 = self.finest_window
        num_windows0 = t // k0
        windows0 = raw_patches.view(b, num_windows0, k0, p, patch_dim).mean(dim=2)
        finest_grid = self.pyramid_proj(windows0)  # (B, num_windows0, P, d)

        ascending_idx = sorted(range(len(self.window_sizes)), key=lambda i: self.window_sizes[i])
        views_ascending = []
        current = finest_grid
        prev_ratio = 1
        for rank, _ in enumerate(ascending_idx):
            ratio = self.pyramid_ratios[rank]
            step = ratio // prev_ratio
            if step > 1:
                nw, p_, d = current.shape[1], current.shape[2], current.shape[3]
                current = current.view(b, nw // step, step, p_, d).mean(dim=2)
            views_ascending.append(current)
            prev_ratio = ratio

        outputs: list[torch.Tensor | None] = [None] * len(self.window_sizes)
        for pos, view_idx in enumerate(ascending_idx):
            grid = views_ascending[pos]
            outputs[view_idx] = grid.reshape(b, -1, grid.shape[-1])
        return outputs

    def forward(self, clip: torch.Tensor, x_bar: torch.Tensor) -> list[torch.Tensor]:
        """Returns gated tokens Z~_s (one tensor per view), each (B, N_s, d)."""
        raw_patches = patchify(clip, self.patch_size)

        if self.view_mode == "pyramid":
            tokens = self._pyramid_forward(raw_patches)
        else:
            tokens = [embed(raw_patches, x_bar) for embed in self.view_embeds]

        alpha = self.scale_gate(x_bar)  # (B, S)
        gated = [alpha[:, s : s + 1].unsqueeze(-1) * tokens[s] for s in range(len(tokens))]
        return gated
