import dataclasses

import pytest
import torch

from crowdvit.models.stem import Conv3DStem
from crowdvit.models.tubelet import MultiScaleTubeletEmbed


def _expected_token_counts(cfg):
    grid = (cfg.image_size // cfg.patch_size) ** 2
    return [(cfg.num_frames // k) * grid for k in cfg.view_window_sizes]


@pytest.mark.parametrize("view_mode", ["adaptive", "fixed", "pyramid"])
def test_token_counts_match_paper_formula(tiny_model_cfg, tiny_clip, view_mode):
    cfg = dataclasses.replace(tiny_model_cfg, view_mode=view_mode)
    stem = Conv3DStem(cfg)
    tubelet = MultiScaleTubeletEmbed(cfg)

    x_bar = stem(tiny_clip)
    views = tubelet(tiny_clip, x_bar)

    expected = _expected_token_counts(cfg)
    assert [v.shape[1] for v in views] == expected
    assert all(v.shape[0] == tiny_clip.shape[0] for v in views)
    assert all(v.shape[2] == cfg.hidden_dim for v in views)


def test_paper_reference_configuration_token_counts():
    """T=16, H=W=224, patch=16, windows=[8,4,2] -> N=[392, 784, 1568] (paper's numbers)."""
    from crowdvit.config import ModelConfig

    cfg = ModelConfig()  # defaults match the paper's implementation section
    assert _expected_token_counts(cfg) == [392, 784, 1568]


def test_scale_gate_sums_to_one(tiny_model_cfg, tiny_clip):
    stem = Conv3DStem(tiny_model_cfg)
    tubelet = MultiScaleTubeletEmbed(tiny_model_cfg)
    x_bar = stem(tiny_clip)
    alpha = tubelet.scale_gate(x_bar)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(tiny_clip.shape[0]), atol=1e-5)
