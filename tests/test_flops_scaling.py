"""Verifies the paper's central efficiency claim at the attention-module
level: Performer attention cost grows ~linearly with token count while full
softmax attention grows ~quadratically (Eq. "complexity_compact")."""

import torch

from crowdvit.models.performer_attention import FullSelfAttention, PerformerSelfAttention
from crowdvit.utils.flops import count_gflops


def _gflops_for_seq_len(attn, seq_len, dim):
    x = torch.randn(1, seq_len, dim)
    return count_gflops(attn, x)


def test_performer_attention_scales_linearly_with_tokens():
    attn = PerformerSelfAttention(dim=64, num_heads=4, kernel_dim=32)
    small = _gflops_for_seq_len(attn, 200, dim=64)
    large = _gflops_for_seq_len(attn, 800, dim=64)
    ratio = large / small
    # tokens grew 4x; linear-attention cost should grow ~4x, not ~16x.
    assert 3.0 < ratio < 6.0


def test_full_attention_scales_quadratically_with_tokens():
    attn = FullSelfAttention(dim=64, num_heads=4)
    small = _gflops_for_seq_len(attn, 200, dim=64)
    large = _gflops_for_seq_len(attn, 800, dim=64)
    ratio = large / small
    # tokens grew 4x; quadratic attention cost should grow ~16x.
    assert ratio > 10.0


def test_performer_is_cheaper_than_full_at_large_seq_len():
    seq_len = 1000
    performer = _gflops_for_seq_len(
        PerformerSelfAttention(dim=64, num_heads=4, kernel_dim=32), seq_len, dim=64
    )
    full = _gflops_for_seq_len(FullSelfAttention(dim=64, num_heads=4), seq_len, dim=64)
    assert performer < full
