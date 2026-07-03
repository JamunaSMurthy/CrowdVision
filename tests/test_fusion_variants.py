import dataclasses

import pytest
import torch

from crowdvit.models.fusion import build_fusion

FUSION_TYPES = ["kga", "concat_only", "performer_only", "graph_only", "full_cross_attention"]


@pytest.mark.parametrize("fusion_type", FUSION_TYPES)
def test_fusion_variants_preserve_shapes(tiny_model_cfg, fusion_type):
    cfg = dataclasses.replace(tiny_model_cfg, fusion_type=fusion_type)
    fusion = build_fusion(cfg)
    views = [
        torch.randn(2, 8, cfg.hidden_dim),
        torch.randn(2, 16, cfg.hidden_dim),
    ]
    outputs = fusion(views)
    assert len(outputs) == len(views)
    assert [o.shape for o in outputs] == [v.shape for v in views]
    assert all(torch.isfinite(o).all() for o in outputs)


def test_concat_only_is_identity(tiny_model_cfg):
    cfg = dataclasses.replace(tiny_model_cfg, fusion_type="concat_only")
    fusion = build_fusion(cfg)
    views = [torch.randn(2, 8, cfg.hidden_dim), torch.randn(2, 16, cfg.hidden_dim)]
    outputs = fusion(views)
    for out, view in zip(outputs, views):
        assert torch.equal(out, view)


def test_unknown_fusion_type_raises(tiny_model_cfg):
    cfg = dataclasses.replace(tiny_model_cfg, fusion_type="not_a_real_fusion")
    with pytest.raises(ValueError):
        build_fusion(cfg)
