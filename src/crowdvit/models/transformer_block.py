"""Pre-norm transformer block used by the view encoders and the Global
Fusion Encoder: LayerNorm -> attention -> residual, LayerNorm -> MLP ->
residual (Section 3.3 / 3.5 of the paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdvit.models.performer_attention import build_attention


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: str,
        attn_dropout: float,
        proj_dropout: float,
        mlp_ratio: float,
        seq_len: int | None = None,
        kernel_dim: int = 256,
        linformer_proj_dim: int = 128,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = build_attention(
            attention_type,
            dim,
            num_heads,
            attn_dropout,
            proj_dropout,
            seq_len=seq_len,
            kernel_dim=kernel_dim,
            linformer_proj_dim=linformer_proj_dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(**block_kwargs) for _ in range(depth)])
        self.norm = nn.LayerNorm(block_kwargs["dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def redraw_performer_projections(self) -> None:
        for block in self.blocks:
            if hasattr(block.attn, "redraw_projection_matrix"):
                block.attn.redraw_projection_matrix()
