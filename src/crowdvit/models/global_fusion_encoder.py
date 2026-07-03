"""Global Fusion Encoder (Section 3.5): adds a learned view-type embedding to
every token of its source view, concatenates the KGA-aligned views into one
sequence, refines it with a shallow Performer stack, and pools it via
classification-token pooling + average pooling over the remaining tokens.

Kept intentionally separate from KGA: KGA performs cross-view temporal
alignment, this module performs post-fusion representation refinement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from crowdvit.config import ModelConfig
from crowdvit.models.transformer_block import TransformerEncoder


def view_token_counts(cfg: ModelConfig) -> list[int]:
    grid = (cfg.image_size // cfg.patch_size) ** 2
    counts = []
    for k in cfg.view_window_sizes:
        assert cfg.num_frames % k == 0, f"num_frames must be divisible by window size {k}"
        counts.append((cfg.num_frames // k) * grid)
    return counts


class GlobalFusionEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.use_encoder = cfg.use_global_fusion_encoder
        self.view_type_embed = nn.Parameter(torch.zeros(cfg.num_views, cfg.hidden_dim))
        nn.init.trunc_normal_(self.view_type_embed, std=0.02)

        if self.use_encoder:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            total_tokens = sum(view_token_counts(cfg)) + 1
            self.encoder = TransformerEncoder(
                depth=cfg.fusion_encoder_depth,
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                attention_type=cfg.attention_type,
                attn_dropout=cfg.attn_dropout,
                proj_dropout=cfg.proj_dropout,
                mlp_ratio=cfg.mlp_ratio,
                seq_len=total_tokens,
                kernel_dim=cfg.kernel_dim,
                linformer_proj_dim=cfg.linformer_proj_dim,
            )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        tagged = [v + self.view_type_embed[s] for s, v in enumerate(views)]
        merged = torch.cat(tagged, dim=1)  # (B, N_tot, d)

        if not self.use_encoder:
            return merged.mean(dim=1)

        batch_size = merged.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        merged = torch.cat([cls, merged], dim=1)
        merged = self.encoder(merged)

        cls_pooled = merged[:, 0]
        mean_pooled = merged[:, 1:].mean(dim=1)
        return (cls_pooled + mean_pooled) / 2.0
