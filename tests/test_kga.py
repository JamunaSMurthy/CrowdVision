import dataclasses

import pytest
import torch

from crowdvit.models.kga import AffinityGraph, KernelGraphAttention


@pytest.mark.parametrize("affinity_type", ["learnable", "uniform", "random", "fixed", "none"])
def test_affinity_graph_rows_sum_to_one(affinity_type):
    graph = AffinityGraph(dim=16, num_views=3, affinity_type=affinity_type)
    descriptors = torch.randn(4, 3, 16)
    A = graph(descriptors)
    assert A.shape == (4, 3, 3)
    assert torch.allclose(A.sum(dim=-1), torch.ones(4, 3), atol=1e-4)


def test_affinity_none_is_identity():
    graph = AffinityGraph(dim=16, num_views=3, affinity_type="none")
    descriptors = torch.randn(2, 3, 16)
    A = graph(descriptors)
    expected = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(A, expected)


@pytest.mark.parametrize("affinity_type", ["learnable", "uniform", "random", "fixed", "none"])
def test_kga_preserves_per_view_token_counts(tiny_model_cfg, affinity_type):
    cfg = dataclasses.replace(tiny_model_cfg, affinity_type=affinity_type)
    kga = KernelGraphAttention(cfg)
    views = [torch.randn(2, 8, cfg.hidden_dim), torch.randn(2, 16, cfg.hidden_dim)]
    outputs = kga(views)
    assert [o.shape for o in outputs] == [v.shape for v in views]
    assert all(torch.isfinite(o).all() for o in outputs)


def test_kga_none_affinity_matches_pure_self_attention(tiny_model_cfg):
    """With affinity_type='none' (identity graph), each view should only
    aggregate its own kernel summary, i.e. behave like independent Performer
    attention per view rather than mixing across views.
    """
    cfg = dataclasses.replace(tiny_model_cfg, affinity_type="none")
    kga = KernelGraphAttention(cfg)
    views = [torch.randn(2, 8, cfg.hidden_dim), torch.randn(2, 16, cfg.hidden_dim)]

    phi_q0, kv0, ksum0 = kga._project_view(views[0], 0)
    numerator = torch.einsum("bhnr,bhrd->bhnd", phi_q0, kv0)
    denominator = torch.einsum("bhnr,bhr->bhn", phi_q0, ksum0).unsqueeze(-1)
    expected0 = (numerator / (denominator + 1e-6)).permute(0, 2, 1, 3).reshape(2, 8, cfg.hidden_dim)
    expected0 = kga.out_proj[0](expected0)

    kga.eval()
    with torch.no_grad():
        outputs = kga(views)
    assert torch.allclose(outputs[0], expected0, atol=1e-5)
