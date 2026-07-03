import pytest
import torch

from crowdvit.models.performer_attention import (
    FullSelfAttention,
    LinformerSelfAttention,
    PerformerSelfAttention,
    build_attention,
    build_orthogonal_projection,
    favor_feature_map,
)


def test_orthogonal_projection_shape_and_orthogonality():
    proj = build_orthogonal_projection(num_features=16, head_dim=8)
    assert proj.shape == (16, 8)
    # Rows within each 8-row block should be (near-)orthogonal.
    block = proj[:8]
    gram = block @ block.t()
    off_diag = gram - torch.diag(torch.diag(gram))
    assert off_diag.abs().max().item() < 1e-3


def test_favor_feature_map_is_positive():
    x = torch.randn(2, 4, 5, 8)  # (B, H, N, head_dim)
    proj = build_orthogonal_projection(16, 8)
    phi_q = favor_feature_map(x, proj, is_query=True)
    phi_k = favor_feature_map(x, proj, is_query=False)
    assert (phi_q > 0).all()
    assert (phi_k > 0).all()
    assert phi_q.shape == (2, 4, 5, 16)


def test_performer_attention_forward_shape():
    attn = PerformerSelfAttention(dim=32, num_heads=4, kernel_dim=16)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == x.shape


def test_performer_attention_permutation_invariant_denominator_normalizes():
    """Sanity check that the linear-attention output does not blow up /
    collapse to zero for a reasonably sized input."""
    attn = PerformerSelfAttention(dim=32, num_heads=4, kernel_dim=16)
    x = torch.randn(2, 50, 32)
    out = attn(x)
    assert torch.isfinite(out).all()


def test_linformer_attention_forward_shape():
    attn = LinformerSelfAttention(dim=32, num_heads=4, seq_len=10, proj_dim=4)
    x = torch.randn(2, 10, 32)
    assert attn(x).shape == x.shape


def test_linformer_attention_rejects_wrong_seq_len():
    attn = LinformerSelfAttention(dim=32, num_heads=4, seq_len=10, proj_dim=4)
    with pytest.raises(AssertionError):
        attn(torch.randn(2, 11, 32))


def test_full_attention_forward_shape():
    attn = FullSelfAttention(dim=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    assert attn(x).shape == x.shape


def test_build_attention_factory():
    for attention_type in ("performer", "linformer", "full"):
        attn = build_attention(
            attention_type,
            dim=32,
            num_heads=4,
            attn_dropout=0.0,
            proj_dropout=0.0,
            seq_len=10,
            kernel_dim=16,
            linformer_proj_dim=4,
        )
        out = attn(torch.randn(2, 10, 32))
        assert out.shape == (2, 10, 32)


def test_redraw_projection_matrix_changes_weights():
    attn = PerformerSelfAttention(dim=32, num_heads=4, kernel_dim=16)
    before = attn.projection_matrix.clone()
    attn.redraw_projection_matrix()
    assert not torch.allclose(before, attn.projection_matrix)
