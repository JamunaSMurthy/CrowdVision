"""Full CrowdViT model: Conv3D stem -> adaptive multi-scale tubelet views ->
per-view Performer encoders -> Kernel-Graph Attention -> Global Fusion
Encoder -> MLP head (Algorithm 1 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from crowdvit.config import ModelConfig
from crowdvit.models.fusion import build_fusion
from crowdvit.models.global_fusion_encoder import GlobalFusionEncoder, view_token_counts
from crowdvit.models.head import MLPHead
from crowdvit.models.stem import Conv3DStem
from crowdvit.models.transformer_block import TransformerEncoder
from crowdvit.models.tubelet import MultiScaleTubeletEmbed


@dataclass
class CrowdViTOutput:
    logits: torch.Tensor
    view_descriptors: list[torch.Tensor]  # post-fusion per-view descriptors, for L_view
    frame_sequence: torch.Tensor  # per-window descriptors of the finest view, for L_temp
    pooled_feature: torch.Tensor


class CrowdViT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = Conv3DStem(cfg)
        self.tubelet = MultiScaleTubeletEmbed(cfg)

        seq_lens = view_token_counts(cfg)
        self.view_encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    depth=cfg.view_encoder_depth,
                    dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    attention_type=cfg.attention_type,
                    attn_dropout=cfg.attn_dropout,
                    proj_dropout=cfg.proj_dropout,
                    mlp_ratio=cfg.mlp_ratio,
                    seq_len=seq_lens[s],
                    kernel_dim=cfg.kernel_dim,
                    linformer_proj_dim=cfg.linformer_proj_dim,
                )
                for s in range(cfg.num_views)
            ]
        )

        self.fusion = build_fusion(cfg)
        self.global_fusion = GlobalFusionEncoder(cfg)
        self.head = MLPHead(cfg)

        self._finest_idx = min(range(cfg.num_views), key=lambda i: cfg.view_window_sizes[i])
        self._finest_window = cfg.view_window_sizes[self._finest_idx]
        self._grid_size = (cfg.image_size // cfg.patch_size) ** 2

    def forward(self, clip: torch.Tensor) -> CrowdViTOutput:
        """clip: (B, T, C, H, W) -> CrowdViTOutput."""
        x_bar = self.stem(clip)
        gated_views = self.tubelet(clip, x_bar)

        encoded_views = [encoder(v) for encoder, v in zip(self.view_encoders, gated_views)]
        fused_views = self.fusion(encoded_views)

        pooled_feature = self.global_fusion(fused_views)
        logits = self.head(pooled_feature)

        view_descriptors = [v.mean(dim=1) for v in fused_views]

        finest = encoded_views[self._finest_idx]
        b, _, d = finest.shape
        num_windows = clip.shape[1] // self._finest_window
        frame_sequence = finest.view(b, num_windows, self._grid_size, d).mean(dim=2)

        return CrowdViTOutput(
            logits=logits,
            view_descriptors=view_descriptors,
            frame_sequence=frame_sequence,
            pooled_feature=pooled_feature,
        )

    def redraw_performer_projections(self) -> None:
        """Redraw all FAVOR+ random projection matrices in place. Intended to
        be called periodically during training (see engine/trainer.py) as a
        variance-reduction measure recommended for Performer-style models.
        """
        for encoder in self.view_encoders:
            encoder.redraw_performer_projections()
        if hasattr(self.global_fusion, "encoder"):
            self.global_fusion.encoder.redraw_performer_projections()
        if hasattr(self.fusion, "redraw_projection_matrix"):
            self.fusion.redraw_projection_matrix()
        if hasattr(self.fusion, "attn") and hasattr(self.fusion.attn, "redraw_projection_matrix"):
            self.fusion.attn.redraw_projection_matrix()
