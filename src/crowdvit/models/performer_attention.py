"""Performer (FAVOR+) linear attention (Eq. 5 of the paper), plus the
Linformer and full-softmax attention variants used only for the "Attention
mechanism" ablation table.

The FAVOR+ feature map approximates the softmax kernel with positive
orthogonal random features (Choromanski et al., 2020): each query/key vector
is mapped through a fixed random projection so that dot products of the
mapped vectors are, in expectation, equal to the softmax kernel of the
original vectors. This lets attention be reordered as
``phi(Q)(phi(K)^T V) / phi(Q)(phi(K)^T 1)``, which costs O(N r d) instead of
the O(N^2 d) of dense softmax attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _orthogonal_block(size: int, device=None, dtype=None) -> torch.Tensor:
    unstructured = torch.randn((size, size), device=device, dtype=dtype)
    q, _ = torch.linalg.qr(unstructured)
    return q.t()


def build_orthogonal_projection(
    num_features: int, head_dim: int, device=None, dtype=None
) -> torch.Tensor:
    """Orthogonal random features matrix of shape (num_features, head_dim).

    Rows are grouped into orthogonal blocks of size head_dim (via QR of a
    random Gaussian matrix) and rescaled by chi-distributed norms, so the
    marginal distribution of each row matches an iid standard Gaussian while
    rows within a block are mutually orthogonal (lower-variance kernel
    estimate than plain iid Gaussian projections).
    """
    num_full_blocks = num_features // head_dim
    blocks = [_orthogonal_block(head_dim, device, dtype) for _ in range(num_full_blocks)]
    remaining = num_features - num_full_blocks * head_dim
    if remaining > 0:
        blocks.append(_orthogonal_block(head_dim, device, dtype)[:remaining])
    matrix = torch.cat(blocks, dim=0)
    row_norms = torch.randn((num_features, head_dim), device=device, dtype=dtype).norm(dim=1)
    return row_norms.unsqueeze(1) * matrix


def favor_feature_map(
    x: torch.Tensor, projection_matrix: torch.Tensor, is_query: bool, eps: float = 1e-4
) -> torch.Tensor:
    """x: (..., N, head_dim), projection_matrix: (r, head_dim) -> (..., N, r)."""
    head_dim = x.shape[-1]
    num_features = projection_matrix.shape[0]
    data_normalizer = head_dim**-0.25
    ratio = num_features**-0.5

    proj = torch.einsum("...nd,rd->...nr", x * data_normalizer, projection_matrix)
    sq_norm = ((x * data_normalizer) ** 2).sum(dim=-1, keepdim=True) / 2.0

    if is_query:
        stabilizer = proj.amax(dim=-1, keepdim=True)
    else:
        stabilizer = proj.amax(dim=(-2, -1), keepdim=True)

    return ratio * (torch.exp(proj - sq_norm - stabilizer) + eps)


class PerformerSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_dim: int,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_dim = kernel_dim

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.register_buffer(
            "projection_matrix",
            build_orthogonal_projection(kernel_dim, self.head_dim),
            persistent=False,
        )

    @torch.no_grad()
    def redraw_projection_matrix(self) -> None:
        self.projection_matrix = build_orthogonal_projection(
            self.kernel_dim, self.head_dim, device=self.projection_matrix.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)

        phi_q = favor_feature_map(q, self.projection_matrix, is_query=True)
        phi_k = favor_feature_map(k, self.projection_matrix, is_query=False)

        kv = torch.einsum("bhnr,bhnd->bhrd", phi_k, v)
        k_sum = phi_k.sum(dim=2)
        numerator = torch.einsum("bhnr,bhrd->bhnd", phi_q, kv)
        denominator = torch.einsum("bhnr,bhr->bhn", phi_q, k_sum).unsqueeze(-1)
        out = numerator / (denominator + 1e-6)

        out = self.attn_dropout(out)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.proj_dropout(self.out_proj(out))


class LinformerSelfAttention(nn.Module):
    """Ablation baseline (Table: Attention mechanism, "Linformer" row)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        proj_dim: int,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Parameter(torch.randn(seq_len, proj_dim) / seq_len**0.5)
        self.value_proj = nn.Parameter(torch.randn(seq_len, proj_dim) / seq_len**0.5)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        assert n == self.seq_len, f"LinformerSelfAttention expects seq_len={self.seq_len}, got {n}"
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_proj = torch.einsum("nk,bhnd->bhkd", self.key_proj, k)
        v_proj = torch.einsum("nk,bhnd->bhkd", self.value_proj, v)

        attn = torch.einsum("bhnd,bhkd->bhnk", q, k_proj) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhnk,bhkd->bhnd", attn, v_proj)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.proj_dropout(self.out_proj(out))


class FullSelfAttention(nn.Module):
    """Ablation baseline (Table: Attention mechanism, "Full self-attention" row)."""

    def __init__(
        self, dim: int, num_heads: int, attn_dropout: float = 0.1, proj_dropout: float = 0.1
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.proj_dropout(self.out_proj(out))


def build_attention(
    attention_type: str,
    dim: int,
    num_heads: int,
    attn_dropout: float,
    proj_dropout: float,
    seq_len: int | None = None,
    kernel_dim: int = 256,
    linformer_proj_dim: int = 128,
) -> nn.Module:
    if attention_type == "performer":
        return PerformerSelfAttention(dim, num_heads, kernel_dim, attn_dropout, proj_dropout)
    if attention_type == "linformer":
        assert seq_len is not None, "Linformer attention requires a fixed seq_len"
        return LinformerSelfAttention(
            dim, num_heads, seq_len, linformer_proj_dim, attn_dropout, proj_dropout
        )
    if attention_type == "full":
        return FullSelfAttention(dim, num_heads, attn_dropout, proj_dropout)
    raise ValueError(f"Unknown attention_type '{attention_type}'")
